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


def _ensure_db_pool():
    global db_pool, _pool_initialized

    with _pool_lock:
        if _pool_initialized:
            return db_pool

        _pool_initialized = True
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
        except Exception as err:
            print(f"[DB] Pool error: {err}")
            db_pool = None

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
        if "service_date" in columns:
            cursor.execute(
                """
                INSERT INTO queue_records
                    (service_date, queue_number, short_code, jwt_token, pdf_path,
                     status, expires_at, created_at)
                VALUES (CURDATE(), %s, %s, %s, %s, 'waiting', %s, NOW())
                """,
                (
                    ticket["queue_number"],
                    ticket["short_code"],
                    ticket["jwt_token"],
                    ticket.get("storage_url") or ticket["pdf_path"],
                    ticket["expires_at"],
                ),
            )
            record_id = cursor.lastrowid
        else:
            cursor.execute(
                """
                INSERT INTO queue_records
                    (queue_number, short_code, jwt_token, pdf_path,
                     status, expires_at, created_at)
                VALUES (%s, %s, %s, %s, 'waiting', %s, NOW())
                ON DUPLICATE KEY UPDATE
                    short_code = VALUES(short_code),
                    jwt_token  = VALUES(jwt_token),
                    pdf_path   = VALUES(pdf_path),
                    expires_at = VALUES(expires_at),
                    status     = 'waiting'
                """,
                (
                    ticket["queue_number"],
                    ticket["short_code"],
                    ticket["jwt_token"],
                    ticket.get("storage_url") or ticket["pdf_path"],
                    ticket["expires_at"],
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
