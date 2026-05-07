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


db_pool = None
_pool_initialized = False
_pool_lock = Lock()
_mysql_connector = None
_mysql_errors = None
_mysql_pooling = None
_table_columns_cache: dict[str, set[str]] = {}


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


def update_queue_status(queue_number: int, status: str) -> None:
    pool = _ensure_db_pool()
    if pool is None:
        print(f"[DB] No pool - Q{queue_number:03d} status not persisted")
        return

    conn = cursor = None
    try:
        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE queue_records SET status=%s, served_at=NOW() "
            "WHERE queue_number=%s AND status='waiting' "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (status, queue_number),
        )
        conn.commit()
        if cursor.rowcount:
            print(f"[DB] Q{queue_number:03d} status='{status}' persisted")
        else:
            print(f"[DB] Q{queue_number:03d} status='{status}' had no waiting DB row")
    except Exception as exc:
        if _is_pool_exhausted(exc):
            print(f"[DB] Pool exhausted - Q{queue_number:03d} status not updated")
        else:
            print(f"[DB] Error updating Q{queue_number:03d}: {exc}")
    finally:
        close_db_resources(cursor, conn)
