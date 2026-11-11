from backend.config import DB_PATH
from backend.db import apply_migrations


def run_migrations(db_path=DB_PATH):
    """Run idempotent SQLite migrations.

    Kept as a small versioned entry point so scripts/tests do not need to know
    the internals of backend.db. The actual migration SQL remains centralized
    in db.py to avoid two competing schema sources.
    """
    apply_migrations(db_path)


__all__ = ["run_migrations"]
