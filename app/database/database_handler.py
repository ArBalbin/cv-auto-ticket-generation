import json
import time

from fastapi import HTTPException
from threading import Lock, Thread

from core.config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_SSL_CA,
    DB_SSL_MODE,
    DB_USERNAME,
)
from services import object_storage_service


db_pool = None
_pool_initialized = False
_pool_lock = Lock()
_mysql_connector = None
_mysql_errors = None
_mysql_pooling = None
_table_columns_cache: dict[str, set[str]] = {}
_table_exists_cache: dict[str, bool] = {}


def _load_mysql():
    global _mysql_connector, _mysql_errors, _mysql_pooling

    if _mysql_connector is None:
        import mysql.connector as connector
        from mysql.connector import errors as mysql_errors
        from mysql.connector import pooling

        _mysql_connector = connector
        _mysql_errors = mysql_errors
        _mysql_pooling = pooling

    return _mysql_connector, _mysql_errors, _mysql_pooling


def _is_pool_exhausted(exc: Exception) -> bool:
    try:
        _, mysql_errors, _ = _load_mysql()
        return isinstance(exc, mysql_errors.PoolExhausted)
    except Exception:
        return exc.__class__.__name__ == "PoolExhausted"


def _is_operational_error(exc: Exception) -> bool:
    try:
        mysql_connector, _, _ = _load_mysql()
        return isinstance(exc, mysql_connector.OperationalError)
    except Exception:
        return exc.__class__.__name__ == "OperationalError"


def _db_ssl_options() -> dict:
    mode = (DB_SSL_MODE or "DISABLED").upper()
    if mode == "DISABLED":
        return {"ssl_disabled": True}

    options = {"ssl_disabled": False}
    if DB_SSL_CA:
        options["ssl_ca"] = DB_SSL_CA
    if mode in {"VERIFY_CA", "VERIFY_IDENTITY"}:
        options["ssl_verify_cert"] = True
    if mode == "VERIFY_IDENTITY":
        options["ssl_verify_identity"] = True
    return options


# When pool creation fails, wait this long before trying again rather than
# hammering an unreachable server on every request.
_POOL_RETRY_SECONDS = 10.0
_pool_retry_after = 0.0


def _ensure_db_pool():
    """
    Return the connection pool, creating it on first use.

    A failed attempt is retried later instead of being latched forever. The
    previous version set _pool_initialized before trying and never cleared it,
    so a single failure at startup disabled the database for the entire life of
    the process — every later call returned None without reconnecting.

    That is not hypothetical: the hosted backend started while the managed
    database was powered off, and went on reporting "db": false long after the
    database came back. Only a redeploy recovered it. A free-tier host that
    sleeps and wakes, in front of a managed database that can blink, needs to
    be able to reconnect on its own.
    """
    global db_pool, _pool_initialized, _pool_retry_after

    with _pool_lock:
        if _pool_initialized:
            return db_pool
        if time.time() < _pool_retry_after:
            return None

        try:
            _, _, pooling = _load_mysql()
            db_pool = pooling.MySQLConnectionPool(
                pool_name="queueflow_pool",
                pool_size=10,
                pool_reset_session=True,
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USERNAME,
                password=DB_PASSWORD,
                **_db_ssl_options(),
            )
            print("[DB] Pool created")
            # Only latch on success — a failure must stay retryable.
            _pool_initialized = True
        except Exception as err:
            print(f"[DB] Pool error: {err}")
            db_pool = None
            _pool_retry_after = time.time() + _POOL_RETRY_SECONDS

    return db_pool


def warm_up_db_pool() -> None:
    if _pool_initialized:
        return

    Thread(
        target=_ensure_db_pool,
        daemon=True,
        name="DBPoolWarmup",
    ).start()


def get_db_pool():
    return _ensure_db_pool()


def is_database_available(check_connection: bool = False) -> bool:
    if not check_connection:
        return db_pool is not None
    return _ensure_db_pool() is not None


def get_db_connection():
    pool = _ensure_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        return pool.get_connection()
    except Exception as exc:
        if _is_pool_exhausted(exc):
            raise HTTPException(
                status_code=503,
                detail="Database pool exhausted - try again shortly",
            )
        raise HTTPException(status_code=500, detail=str(exc))


def close_db_resources(cursor=None, conn=None) -> None:
    if cursor:
        try:
            cursor.close()
        except Exception:
            pass
    if conn:
        try:
            conn.close()
        except Exception:
            pass


def _get_table_columns(cursor, table_name: str) -> set[str]:
    cached = _table_columns_cache.get(table_name)
    if cached is not None:
        return cached

    cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
    columns = set()
    for row in cursor.fetchall():
        if isinstance(row, dict):
            columns.add(str(row.get("Field", "")))
        elif row:
            columns.add(str(row[0]))

    _table_columns_cache[table_name] = columns
    return columns


def _table_exists(cursor, table_name: str) -> bool:
    cached = _table_exists_cache.get(table_name)
    if cached is not None:
        return cached

    cursor.execute(
        """
        SELECT COUNT(*) AS table_count
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
        """,
        (table_name,),
    )
    row = cursor.fetchone()
    if isinstance(row, dict):
        exists = bool(row.get("table_count"))
    else:
        exists = bool(row and row[0])

    _table_exists_cache[table_name] = exists
    return exists


def _row_value(row, key: str, index: int = 0):
    if isinstance(row, dict):
        return row.get(key)
    if row is None:
        return None
    return row[index]


def _lookup_user_id(cursor, username: str | None) -> int | None:
    username = (username or "").strip()
    if not username or not _table_exists(cursor, "users"):
        return None

    cursor.execute(
        "SELECT id FROM users WHERE username=%s LIMIT 1",
        (username,),
    )
    row = cursor.fetchone()
    value = _row_value(row, "id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _latest_queue_record_id(cursor, queue_number: int) -> int | None:
    if not _table_exists(cursor, "queue_records"):
        return None

    cursor.execute(
        """
        SELECT id
        FROM queue_records
        WHERE queue_number=%s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (queue_number,),
    )
    row = cursor.fetchone()
    value = _row_value(row, "id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_waiting_queue_records(limit: int = 100) -> list[dict]:
    pool = _ensure_db_pool()
    if pool is None:
        return []

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        if not _table_exists(cursor, "queue_records"):
            return []

        columns = _get_table_columns(cursor, "queue_records")
        fields = [
            "id",
            "queue_number",
            "status",
            "created_at",
            "expires_at",
            "served_at",
        ]
        for optional in ("service_date", "short_code", "pdf_path"):
            if optional in columns:
                fields.append(optional)

        where = ["status='waiting'"]
        if "service_date" in columns:
            where.append("service_date=CURDATE()")
        else:
            where.append("created_at >= DATE_SUB(NOW(), INTERVAL 1 DAY)")

        cursor.execute(
            f"""
            SELECT {', '.join(fields)}
            FROM queue_records
            WHERE {' AND '.join(where)}
            ORDER BY created_at ASC, id ASC
            LIMIT %s
            """,
            (max(1, min(500, int(limit))),),
        )
        return list(cursor.fetchall() or [])
    except Exception as exc:
        if _is_pool_exhausted(exc):
            print("[DB] Pool exhausted - waiting ticket list unavailable")
        else:
            print(f"[DB] Error loading waiting tickets: {exc}")
        return []
    finally:
        close_db_resources(cursor, conn)


def _queue_record_context(cursor, queue_record_id: int | None, queue_number: int | None):
    if not queue_record_id:
        return None, queue_number

    columns = _get_table_columns(cursor, "queue_records")
    fields = ["queue_number"]
    if "service_date" in columns:
        fields.insert(0, "service_date")

    cursor.execute(
        f"SELECT {', '.join(fields)} FROM queue_records WHERE id=%s LIMIT 1",
        (queue_record_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None, queue_number

    service_date = _row_value(row, "service_date", 0) if "service_date" in columns else None
    q_index = 1 if "service_date" in columns else 0
    stored_queue_number = _row_value(row, "queue_number", q_index)
    return service_date, stored_queue_number or queue_number


def _insert_queue_event(
    cursor,
    event_type: str,
    queue_record_id: int | None = None,
    queue_number: int | None = None,
    actor_user_id: int | None = None,
    event_note: str | None = None,
) -> None:
    if event_type not in {"created", "served", "no_show", "expired", "reset"}:
        return
    if not _table_exists(cursor, "queue_events"):
        return

    service_date, stored_queue_number = _queue_record_context(
        cursor,
        queue_record_id,
        queue_number,
    )
    cursor.execute(
        """
        INSERT INTO queue_events
            (queue_record_id, service_date, queue_number, event_type,
             actor_user_id, event_note, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """,
        (
            queue_record_id,
            service_date,
            stored_queue_number,
            event_type,
            actor_user_id,
            event_note,
        ),
    )


def save_ticket_record(ticket: dict) -> bool:
    pool = _ensure_db_pool()
    if pool is None:
        print("[TicketWorker] No DB pool - ticket not persisted")
        return False

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        columns = _get_table_columns(cursor, "queue_records")
        record_id = None

        # student_id/is_walkin/linked_via/linked_at only exist on schemas
        # that have run the face-recognition migration — degrade gracefully
        # on an older table rather than failing the whole ticket save.
        has_identity_cols = {"student_id", "is_walkin", "linked_via"} <= columns
        identity_fields = (
            ", student_id, is_walkin, linked_via, linked_at" if has_identity_cols else ""
        )
        identity_placeholders = ", %s, %s, %s, NOW()" if has_identity_cols else ""
        identity_values = (
            (ticket.get("student_id"), ticket.get("is_walkin", True), ticket.get("linked_via", "manual"))
            if has_identity_cols else ()
        )
        identity_update = (
            ", student_id = VALUES(student_id), is_walkin = VALUES(is_walkin), "
            "linked_via = VALUES(linked_via), linked_at = VALUES(linked_at)"
            if has_identity_cols else ""
        )

        if "service_date" in columns:
            cursor.execute(
                f"""
                INSERT INTO queue_records
                    (service_date, queue_number, short_code, jwt_token, pdf_path,
                     status, expires_at, created_at{identity_fields})
                VALUES (CURDATE(), %s, %s, %s, %s, 'waiting', %s, NOW(){identity_placeholders})
                """,
                (
                    ticket["queue_number"],
                    ticket["short_code"],
                    ticket["jwt_token"],
                    ticket.get("storage_url") or ticket["pdf_path"],
                    ticket["expires_at"],
                    *identity_values,
                ),
            )
            record_id = cursor.lastrowid
        else:
            cursor.execute(
                f"""
                INSERT INTO queue_records
                    (queue_number, short_code, jwt_token, pdf_path,
                     status, expires_at, created_at{identity_fields})
                VALUES (%s, %s, %s, %s, 'waiting', %s, NOW(){identity_placeholders})
                ON DUPLICATE KEY UPDATE
                    short_code = VALUES(short_code),
                    jwt_token  = VALUES(jwt_token),
                    pdf_path   = VALUES(pdf_path),
                    expires_at = VALUES(expires_at),
                    status     = 'waiting'{identity_update}
                """,
                (
                    ticket["queue_number"],
                    ticket["short_code"],
                    ticket["jwt_token"],
                    ticket.get("storage_url") or ticket["pdf_path"],
                    ticket["expires_at"],
                    *identity_values,
                ),
            )
            record_id = cursor.lastrowid or _latest_queue_record_id(
                cursor,
                ticket["queue_number"],
            )
        _insert_queue_event(
            cursor,
            event_type="created",
            queue_record_id=record_id,
            queue_number=ticket["queue_number"],
            event_note="Ticket generated by detector",
        )
        conn.commit()
        print(f"[TicketWorker] Q{ticket['queue_number']:03d} saved to DB")
        return True
    except Exception as exc:
        if _is_pool_exhausted(exc):
            print("[TicketWorker] DB pool exhausted")
        elif _is_operational_error(exc):
            print(f"[TicketWorker] DB operational error: {exc}")
        else:
            print(f"[TicketWorker] DB error: {exc}")
    finally:
        close_db_resources(cursor, conn)

    return False


def update_queue_status(
    queue_number: int,
    status: str,
    actor_username: str | None = None,
) -> None:
    if status not in {"served", "no_show", "expired"}:
        print(f"[DB] Ignored unsupported status '{status}' for Q{queue_number:03d}")
        return

    pool = _ensure_db_pool()
    if pool is None:
        print(f"[DB] No pool - Q{queue_number:03d} status not persisted")
        return

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        columns = _get_table_columns(cursor, "queue_records")
        actor_user_id = _lookup_user_id(cursor, actor_username)

        cursor.execute(
            """
            SELECT id, pdf_path
            FROM queue_records
            WHERE queue_number=%s AND status='waiting'
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            FOR UPDATE
            """,
            (queue_number,),
        )
        row = cursor.fetchone()
        record_id = _row_value(row, "id")
        pdf_path = _row_value(row, "pdf_path", 1)
        if not record_id:
            conn.commit()
            print(f"[DB] Q{queue_number:03d} status='{status}' had no waiting DB row")
            return

        set_parts = ["status=%s", "served_at=NOW()"]
        params = [status]
        if status == "served" and "served_by_user_id" in columns:
            set_parts.append("served_by_user_id=%s")
            params.append(actor_user_id)
        params.append(record_id)

        cursor.execute(
            f"UPDATE queue_records SET {', '.join(set_parts)} WHERE id=%s",
            tuple(params),
        )
        updated_rows = cursor.rowcount
        _insert_queue_event(
            cursor,
            event_type=status,
            queue_record_id=int(record_id),
            queue_number=queue_number,
            actor_user_id=actor_user_id,
            event_note=f"Queue marked {status}",
        )
        conn.commit()
        if updated_rows:
            print(f"[DB] Q{queue_number:03d} status='{status}' persisted")
            if status == "served" and pdf_path:
                object_storage_service.delete_ticket_object(str(pdf_path))
        else:
            print(f"[DB] Q{queue_number:03d} status='{status}' had no waiting DB row")
    except Exception as exc:
        if _is_pool_exhausted(exc):
            print(f"[DB] Pool exhausted - Q{queue_number:03d} status not updated")
        else:
            print(f"[DB] Error updating Q{queue_number:03d}: {exc}")
    finally:
        close_db_resources(cursor, conn)


def record_queue_reset(actor_username: str | None = None) -> None:
    pool = _ensure_db_pool()
    if pool is None:
        return

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        actor_user_id = _lookup_user_id(cursor, actor_username)
        _insert_queue_event(
            cursor,
            event_type="reset",
            actor_user_id=actor_user_id,
            event_note="Queue reset by staff",
        )
        conn.commit()
    except Exception as exc:
        print(f"[DB] Error recording queue reset: {exc}")
    finally:
        close_db_resources(cursor, conn)


def record_counter_config_change(
    old_counters: int | None,
    new_counters: int,
    avg_service_time: float,
    actor_username: str | None = None,
) -> None:
    if old_counters == new_counters:
        return

    pool = _ensure_db_pool()
    if pool is None:
        return

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        if not _table_exists(cursor, "counter_config_history"):
            return
        actor_user_id = _lookup_user_id(cursor, actor_username)
        cursor.execute(
            """
            INSERT INTO counter_config_history
                (old_counters, new_counters, avg_service_time,
                 changed_by_user_id, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            """,
            (
                old_counters,
                new_counters,
                avg_service_time,
                actor_user_id,
            ),
        )
        conn.commit()
        print(f"[DB] Counter config change persisted: {old_counters} -> {new_counters}")
    except Exception as exc:
        print(f"[DB] Error recording counter config change: {exc}")
    finally:
        close_db_resources(cursor, conn)


def measure_avg_service_time(num_counters: int, window_minutes: int = 120,
                              min_samples: int = 5) -> float | None:
    """Estimate avg service time (minutes/person/counter) from recent served records.

    Uses consecutive inter-departure gaps from served_at timestamps.
    With c parallel counters all busy: inter_departure ≈ avg_service_time / c,
    so avg_service_time = mean(gap) * c.

    Gaps > 15 min are excluded (idle counter, not a service completion).
    Returns None when there are fewer than min_samples valid gaps.
    """
    pool = _ensure_db_pool()
    if pool is None:
        return None

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        if not _table_exists(cursor, "queue_records"):
            return None

        cursor.execute(
            """
            SELECT served_at
            FROM queue_records
            WHERE status = 'served'
              AND served_at IS NOT NULL
              AND served_at >= DATE_SUB(NOW(), INTERVAL %s MINUTE)
            ORDER BY served_at ASC
            """,
            (window_minutes,),
        )
        rows = cursor.fetchall() or []
        timestamps = [
            row["served_at"].timestamp() if hasattr(row["served_at"], "timestamp")
            else float(row["served_at"])
            for row in rows
            if row.get("served_at") is not None
        ]

        if len(timestamps) < min_samples + 1:
            return None

        gaps = []
        for i in range(1, len(timestamps)):
            delta_min = (timestamps[i] - timestamps[i - 1]) / 60.0
            if 0.1 <= delta_min <= 15.0:
                gaps.append(delta_min)

        if len(gaps) < min_samples:
            return None

        measured = (sum(gaps) / len(gaps)) * max(1, num_counters)
        measured = max(0.5, min(measured, 30.0))
        return round(measured, 2)

    except Exception as exc:
        print(f"[DB] measure_avg_service_time error: {exc}")
        return None
    finally:
        close_db_resources(cursor, conn)


# STUDENT SELF-REGISTRATION + FACE ENROLLMENT (Algorithm 4 support)
#
# Accounts are created via self-service Sign-in-with-Google against the
# student's NCF Gbox account — there is no staff-operated enrollment path
# and no password is ever stored here (Google is the credential authority).
# Registration is two steps: create_student_account() at first Google
# sign-in, then save_student_face_embedding() once the student captures
# their face in the app. Only a derived embedding vector is ever persisted
# — never a face image — matching the system's existing "no raw video
# stored" stance, extended to biometric data.

def _student_id_by_school_id(cursor, school_id: str) -> int | None:
    cursor.execute(
        "SELECT id FROM student_profiles WHERE school_id=%s LIMIT 1",
        (school_id,),
    )
    row = cursor.fetchone()
    value = _row_value(row, "id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_STUDENT_PROFILE_FIELDS = (
    "id, school_id, gbox_email, google_sub, full_name, is_active, "
    "num_enrollment_samples, enrolled_at, last_login, face_enrolled_at"
)


def get_student_by_google_sub(google_sub: str) -> dict | None:
    """Login lookup — google_sub is Google's stable per-account identifier."""
    pool = _ensure_db_pool()
    if pool is None:
        return None

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"SELECT {_STUDENT_PROFILE_FIELDS} FROM student_profiles "
            f"WHERE google_sub=%s LIMIT 1",
            (google_sub,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[DB] Error fetching student by google_sub: {exc}")
        return None
    finally:
        close_db_resources(cursor, conn)


def get_student_by_id(student_id: int) -> dict | None:
    pool = _ensure_db_pool()
    if pool is None:
        return None

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"SELECT {_STUDENT_PROFILE_FIELDS} FROM student_profiles "
            f"WHERE id=%s LIMIT 1",
            (student_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[DB] Error fetching student id={student_id}: {exc}")
        return None
    finally:
        close_db_resources(cursor, conn)


def get_student_by_gbox_email(gbox_email: str) -> dict | None:
    pool = _ensure_db_pool()
    if pool is None:
        return None

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"SELECT {_STUDENT_PROFILE_FIELDS} FROM student_profiles "
            f"WHERE gbox_email=%s LIMIT 1",
            (gbox_email,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[DB] Error fetching student by gbox_email: {exc}")
        return None
    finally:
        close_db_resources(cursor, conn)


def create_student_account(
    school_id: str,
    gbox_email: str,
    google_sub: str,
    full_name: str | None = None,
) -> int | None:
    """First-time self-registration. No embedding yet — face capture is a
    separate later step (save_student_face_embedding)."""
    pool = _ensure_db_pool()
    if pool is None:
        print("[DB] No pool - student account not created")
        return None

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO student_profiles
                (school_id, gbox_email, google_sub, full_name, enrolled_at)
            VALUES (%s, %s, %s, %s, NOW())
            """,
            (school_id, gbox_email, google_sub, full_name),
        )
        conn.commit()
        student_id = cursor.lastrowid
        print(f"[DB] Student account created: school_id={school_id}")
        return student_id
    except Exception as exc:
        if exc.__class__.__name__ == "IntegrityError":
            print(f"[DB] Student account already exists (school_id={school_id}): {exc}")
        else:
            print(f"[DB] Error creating student account ({school_id}): {exc}")
        return None
    finally:
        close_db_resources(cursor, conn)


def save_student_face_embedding(
    student_id: int,
    embedding_json: list,
    embedding_model_version: str,
    num_samples: int,
) -> bool:
    pool = _ensure_db_pool()
    if pool is None:
        print("[DB] No pool - face embedding not saved")
        return False

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE student_profiles
            SET embedding               = %s,
                embedding_model_version = %s,
                num_enrollment_samples  = %s,
                face_enrolled_at        = NOW(),
                is_active               = TRUE,
                updated_at              = NOW()
            WHERE id=%s
            """,
            (json.dumps(embedding_json), embedding_model_version, num_samples, student_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            print(f"[DB] Face embedding saved for student_id={student_id}")
        return updated
    except Exception as exc:
        print(f"[DB] Error saving face embedding (student_id={student_id}): {exc}")
        return False
    finally:
        close_db_resources(cursor, conn)


def touch_student_last_login(student_id: int) -> None:
    pool = _ensure_db_pool()
    if pool is None:
        return

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE student_profiles SET last_login=NOW() WHERE id=%s",
            (student_id,),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        close_db_resources(cursor, conn)


def get_student_by_school_id(school_id: str) -> dict | None:
    pool = _ensure_db_pool()
    if pool is None:
        return None

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"SELECT {_STUDENT_PROFILE_FIELDS} FROM student_profiles "
            f"WHERE school_id=%s LIMIT 1",
            (school_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[DB] Error fetching student {school_id}: {exc}")
        return None
    finally:
        close_db_resources(cursor, conn)


def list_students() -> list[dict]:
    """Roster for the staff dashboard. Never returns embeddings."""
    pool = _ensure_db_pool()
    if pool is None:
        return []

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"SELECT {_STUDENT_PROFILE_FIELDS} FROM student_profiles "
            f"ORDER BY enrolled_at DESC"
        )
        return list(cursor.fetchall() or [])
    except Exception as exc:
        print(f"[DB] Error listing students: {exc}")
        return []
    finally:
        close_db_resources(cursor, conn)


def delete_student_profile(student_id: int) -> bool:
    """Hard delete — supports Data Privacy Act erasure requests."""
    pool = _ensure_db_pool()
    if pool is None:
        return False

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM student_profiles WHERE id=%s", (student_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            print(f"[DB] Student profile {student_id} deleted")
        return deleted
    except Exception as exc:
        print(f"[DB] Error deleting student {student_id}: {exc}")
        return False
    finally:
        close_db_resources(cursor, conn)


def get_all_active_embeddings() -> list[tuple]:
    """(student_id, embedding_as_list) pairs for every active enrollment.

    The caller (face_service.match_student, via queue_tracker) is
    responsible for excluding students already linked today — see
    is_student_already_queued_today().
    """
    pool = _ensure_db_pool()
    if pool is None:
        return []

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, embedding FROM student_profiles WHERE is_active=TRUE"
        )
        rows = cursor.fetchall() or []
        result = []
        for row in rows:
            try:
                embedding = json.loads(row["embedding"])
            except (TypeError, ValueError):
                continue
            result.append((int(row["id"]), embedding))
        return result
    except Exception as exc:
        print(f"[DB] Error loading student embeddings: {exc}")
        return []
    finally:
        close_db_resources(cursor, conn)


def is_student_already_queued_today(student_id: int) -> bool:
    """True once a student has been linked to a queue number today —
    the done-blacklist equivalent: don't re-link the same student twice."""
    pool = _ensure_db_pool()
    if pool is None:
        return False

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        columns = _get_table_columns(cursor, "queue_records")
        if "service_date" in columns:
            cursor.execute(
                "SELECT 1 FROM queue_records WHERE student_id=%s AND service_date=CURDATE() LIMIT 1",
                (student_id,),
            )
        else:
            cursor.execute(
                "SELECT 1 FROM queue_records WHERE student_id=%s AND created_at >= CURDATE() LIMIT 1",
                (student_id,),
            )
        return cursor.fetchone() is not None
    except Exception as exc:
        print(f"[DB] Error checking today's queue state for student {student_id}: {exc}")
        return False
    finally:
        close_db_resources(cursor, conn)


def record_face_match_event(
    student_id: int | None,
    track_id: int | None,
    matched_score: float | None,
    margin: float | None,
    accepted: bool,
) -> None:
    pool = _ensure_db_pool()
    if pool is None:
        return

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO face_match_events
                (student_id, track_id, matched_score, margin, accepted, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            """,
            (student_id, track_id, matched_score, margin, accepted),
        )
        conn.commit()
    except Exception as exc:
        print(f"[DB] Error recording face match event: {exc}")
    finally:
        close_db_resources(cursor, conn)


def record_recognition_metric(metric: dict) -> None:
    """
    One row per successful link — how long recognition took end to end, and
    how many camera frames were spent getting there (see
    database_sql/2026_09_recognition_metrics_migration.sql).

    Instrumentation must never be able to break the queue itself, so every
    failure here is swallowed after logging.
    """
    pool = _ensure_db_pool()
    if pool is None:
        return

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO recognition_metrics
                (track_id, student_id, queue_number, linked_via,
                 first_seen_at, confirmed_at, linked_at,
                 seconds_to_confirm, seconds_to_link,
                 embed_attempts, embed_successes, match_score, match_margin)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                metric.get("track_id"),
                metric.get("student_id"),
                metric.get("queue_number"),
                metric.get("linked_via"),
                metric.get("first_seen_at"),
                metric.get("confirmed_at"),
                metric.get("linked_at"),
                metric.get("seconds_to_confirm"),
                metric.get("seconds_to_link"),
                metric.get("embed_attempts", 0),
                metric.get("embed_successes", 0),
                metric.get("match_score"),
                metric.get("match_margin"),
            ),
        )
        conn.commit()
    except Exception as exc:
        print(f"[DB] Error recording recognition metric: {exc}")
    finally:
        close_db_resources(cursor, conn)
