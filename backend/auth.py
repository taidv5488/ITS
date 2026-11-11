"""Xác thực người dùng: lưu tài khoản trong SQLite, mật khẩu băm PBKDF2-SHA256 + salt."""
import hashlib
import hmac
import secrets

from backend.db import _connect

PBKDF2_ITERATIONS = 200_000

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    full_name TEXT,
    role TEXT NOT NULL DEFAULT 'staff'
)
"""

# Tài khoản mặc định (chỉ tạo khi bảng users còn trống)
DEFAULT_USERS = [
    ("admin", "Admin@123", "Quản trị viên", "admin"),
    ("baove", "Baove@123", "Nhân viên bảo vệ", "staff"),
]

ROLE_LABELS = {"admin": "Quản trị viên", "staff": "Nhân viên"}


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS).hex()


def _insert_user(conn, username: str, password: str, full_name: str, role: str):
    salt = secrets.token_bytes(16)
    conn.execute(
        "INSERT OR REPLACE INTO users(username, password_hash, salt, full_name, role) VALUES (?, ?, ?, ?, ?)",
        (username.lower(), _hash_password(password, salt), salt.hex(), full_name, role),
    )


def init_users():
    with _connect() as conn:
        conn.execute(USERS_SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            for username, password, full_name, role in DEFAULT_USERS:
                _insert_user(conn, username, password, full_name, role)


def verify_user(username: str, password: str) -> dict | None:
    """Trả về thông tin user nếu đúng tài khoản/mật khẩu, ngược lại trả về None."""
    username = (username or "").strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT username, password_hash, salt, full_name, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        _hash_password(password or "", b"0" * 16)  # giữ thời gian xử lý gần như nhau
        return None
    stored_hash, salt_hex = row[1], row[2]
    candidate = _hash_password(password or "", bytes.fromhex(salt_hex))
    if not hmac.compare_digest(candidate, stored_hash):
        return None
    return {"username": row[0], "full_name": row[3] or row[0], "role": row[4]}
