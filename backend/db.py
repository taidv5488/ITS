import json
import sqlite3
from contextlib import closing

import pandas as pd

from backend.config import DB_PATH, SCHEMA_VERSION


PARKING_LOGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS parking_logs (
    ticket_id TEXT PRIMARY KEY,
    plate_entry TEXT,
    plate_exit TEXT,
    entry_time TEXT,
    exit_time TEXT,
    status TEXT,
    match_result TEXT,
    entry_image_path TEXT,
    entry_detected_path TEXT,
    entry_plate_path TEXT,
    exit_image_path TEXT,
    exit_detected_path TEXT,
    exit_plate_path TEXT,
    entry_ocr_confidence REAL,
    exit_ocr_confidence REAL,
    entry_ocr_engine TEXT,
    exit_ocr_engine TEXT,
    entry_ocr_text TEXT,
    exit_ocr_text TEXT,
    entry_manual_override INTEGER NOT NULL DEFAULT 0,
    exit_manual_override INTEGER NOT NULL DEFAULT 0,
    review_required INTEGER NOT NULL DEFAULT 0,
    review_reason TEXT,
    resolved_by TEXT,
    resolved_at TEXT,
    resolution_note TEXT,
    cancel_reason TEXT,
    entry_actor TEXT,
    exit_actor TEXT
)
"""

SCHEMA_MIGRATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    failed_login_count INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    last_login_at TEXT
)
"""

AUDIT_LOGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_username TEXT,
    actor_role TEXT,
    action TEXT NOT NULL,
    target_type TEXT,
    target_id TEXT,
    details_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

PARKING_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS parking_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT,
    event_type TEXT NOT NULL,
    actor_username TEXT,
    actor_role TEXT,
    plate_text TEXT,
    confidence REAL,
    source TEXT,
    details_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


def _connect(db_path=DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _table_exists(cursor, table_name: str) -> bool:
    row = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _get_table_columns(cursor, table_name: str) -> set[str]:
    if not _table_exists(cursor, table_name):
        return set()
    rows = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _ensure_columns(cursor, table_name: str, expected_columns: dict[str, str]):
    existing_columns = _get_table_columns(cursor, table_name)
    for column_name, column_type in expected_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _ensure_parking_logs_columns(cursor):
    _ensure_columns(
        cursor,
        "parking_logs",
        {
            "ticket_id": "TEXT PRIMARY KEY",
            "plate_entry": "TEXT",
            "plate_exit": "TEXT",
            "entry_time": "TEXT",
            "exit_time": "TEXT",
            "status": "TEXT",
            "match_result": "TEXT",
            "entry_image_path": "TEXT",
            "entry_detected_path": "TEXT",
            "entry_plate_path": "TEXT",
            "exit_image_path": "TEXT",
            "exit_detected_path": "TEXT",
            "exit_plate_path": "TEXT",
            "entry_ocr_confidence": "REAL",
            "exit_ocr_confidence": "REAL",
            "entry_ocr_engine": "TEXT",
            "exit_ocr_engine": "TEXT",
            "entry_ocr_text": "TEXT",
            "exit_ocr_text": "TEXT",
            "entry_manual_override": "INTEGER NOT NULL DEFAULT 0",
            "exit_manual_override": "INTEGER NOT NULL DEFAULT 0",
            "review_required": "INTEGER NOT NULL DEFAULT 0",
            "review_reason": "TEXT",
            "resolved_by": "TEXT",
            "resolved_at": "TEXT",
            "resolution_note": "TEXT",
            "cancel_reason": "TEXT",
            "entry_actor": "TEXT",
            "exit_actor": "TEXT",
        },
    )


def _ensure_app_user_columns(cursor):
    _ensure_columns(
        cursor,
        "app_users",
        {
            "is_active": "INTEGER NOT NULL DEFAULT 1",
            "failed_login_count": "INTEGER NOT NULL DEFAULT 0",
            "locked_until": "TEXT",
            "last_login_at": "TEXT",
        },
    )


def apply_migrations(db_path=DB_PATH):
    with _connect(db_path) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(SCHEMA_MIGRATIONS_SCHEMA)
            cursor.execute(PARKING_LOGS_SCHEMA)
            cursor.execute(USERS_SCHEMA)
            cursor.execute(AUDIT_LOGS_SCHEMA)
            cursor.execute(PARKING_EVENTS_SCHEMA)
            _ensure_parking_logs_columns(cursor)
            _ensure_app_user_columns(cursor)
            for version in range(1, SCHEMA_VERSION + 1):
                cursor.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (version,))

            existing_columns = _get_table_columns(cursor, "parking_logs")
            if "ocr_confidence" in existing_columns:
                if "entry_ocr_confidence" in existing_columns:
                    cursor.execute(
                        "UPDATE parking_logs SET entry_ocr_confidence = COALESCE(entry_ocr_confidence, ocr_confidence) WHERE entry_ocr_confidence IS NULL"
                    )
                if "exit_ocr_confidence" in existing_columns:
                    cursor.execute(
                        "UPDATE parking_logs SET exit_ocr_confidence = COALESCE(exit_ocr_confidence, ocr_confidence) WHERE status = 'COMPLETED' AND exit_ocr_confidence IS NULL"
                    )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_parking_logs_ticket_id ON parking_logs(ticket_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_parking_logs_plate_entry ON parking_logs(plate_entry)")
            if {"status", "entry_time"}.issubset(existing_columns):
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_parking_logs_status_entry_time ON parking_logs(status, entry_time DESC)")
            if "entry_time" in existing_columns:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_parking_logs_entry_time ON parking_logs(entry_time DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_parking_events_ticket_time ON parking_events(ticket_id, created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_actor ON audit_logs(actor_username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC)")
        conn.commit()


def init_db(db_path=DB_PATH):
    apply_migrations(db_path)


def fetch_logs(db_path=DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM parking_logs ORDER BY entry_time DESC", conn)


def get_active_tickets(db_path=DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT ticket_id, plate_entry, entry_time, status, review_required, review_reason FROM parking_logs WHERE status IN ('IN_PARKING', 'NEEDS_REVIEW') ORDER BY entry_time DESC",
            conn,
        )


def fetch_ticket_detail(ticket_id: str, db_path=DB_PATH) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM parking_logs WHERE ticket_id = ?", ((ticket_id or "").strip(),)).fetchone()
        return dict(row) if row else None


def fetch_app_user(username: str, db_path=DB_PATH) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT username, password_hash, role, is_active, created_at, failed_login_count, locked_until, last_login_at FROM app_users WHERE username = ?",
            ((username or "").strip(),),
        ).fetchone()
        return dict(row) if row else None


def create_app_user(username: str, password_hash: str, role: str = "viewer", db_path=DB_PATH) -> bool:
    clean_username = (username or "").strip()
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO app_users(username, password_hash, role) VALUES (?, ?, ?)",
            (clean_username, password_hash, role),
        )
        conn.commit()
        return cursor.rowcount > 0


def update_app_user_password_hash(username: str, password_hash: str, db_path=DB_PATH) -> bool:
    clean_username = (username or "").strip()
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE app_users SET password_hash = ? WHERE username = ?",
            (password_hash, clean_username),
        )
        conn.commit()
        return cursor.rowcount > 0


def update_login_success(username: str, db_path=DB_PATH) -> bool:
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE app_users SET failed_login_count = 0, locked_until = NULL, last_login_at = CURRENT_TIMESTAMP WHERE username = ?",
            ((username or "").strip(),),
        )
        conn.commit()
        return cursor.rowcount > 0


def update_login_failure(username: str, locked_until: str | None = None, db_path=DB_PATH) -> bool:
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE app_users
            SET failed_login_count = COALESCE(failed_login_count, 0) + 1,
                locked_until = COALESCE(?, locked_until)
            WHERE username = ?
            """,
            (locked_until, (username or "").strip()),
        )
        conn.commit()
        return cursor.rowcount > 0


def list_app_users(db_path=DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT username, role, is_active, created_at, failed_login_count, locked_until, last_login_at FROM app_users ORDER BY created_at DESC, username ASC",
            conn,
        )


def update_app_user_role(username: str, role: str, db_path=DB_PATH) -> bool:
    clean_username = (username or "").strip()
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE app_users SET role = ? WHERE username = ?", (role, clean_username))
        conn.commit()
        return cursor.rowcount > 0


def update_app_user_active(username: str, is_active: bool, db_path=DB_PATH) -> bool:
    clean_username = (username or "").strip()
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE app_users SET is_active = ? WHERE username = ?", (1 if is_active else 0, clean_username))
        conn.commit()
        return cursor.rowcount > 0


def delete_app_user(username: str, db_path=DB_PATH) -> bool:
    clean_username = (username or "").strip()
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM app_users WHERE username = ?", (clean_username,))
        conn.commit()
        return cursor.rowcount > 0


def append_audit_log(actor_username: str | None, actor_role: str | None, action: str, *, target_type: str | None = None, target_id: str | None = None, details_json: str | None = None, db_path=DB_PATH) -> bool:
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO audit_logs(actor_username, actor_role, action, target_type, target_id, details_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                (actor_username or "").strip() or None,
                (actor_role or "").strip() or None,
                action,
                (target_type or "").strip() or None,
                (target_id or "").strip() or None,
                details_json,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0


def fetch_audit_logs(db_path=DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM audit_logs ORDER BY created_at DESC, id DESC", conn)


def append_parking_event(ticket_id: str | None, event_type: str, *, actor_username: str | None = None, actor_role: str | None = None, plate_text: str | None = None, confidence: float | None = None, source: str | None = None, details: dict | None = None, db_path=DB_PATH) -> bool:
    details_json = json.dumps(details or {}, ensure_ascii=False) if details else None
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO parking_events(ticket_id, event_type, actor_username, actor_role, plate_text, confidence, source, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (ticket_id or "").strip() or None,
                event_type,
                (actor_username or "").strip() or None,
                (actor_role or "").strip() or None,
                plate_text,
                confidence,
                source,
                details_json,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0


def fetch_parking_events(ticket_id: str | None = None, db_path=DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as conn:
        if ticket_id:
            return pd.read_sql_query(
                "SELECT * FROM parking_events WHERE ticket_id = ? ORDER BY created_at DESC, id DESC",
                conn,
                params=((ticket_id or "").strip(),),
            )
        return pd.read_sql_query("SELECT * FROM parking_events ORDER BY created_at DESC, id DESC", conn)
