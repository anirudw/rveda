"""SQLite-backed ICD-10 search utilities for the Rveda environment."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "icd10_mock.json"


def _default_db_path() -> Path:
    env_path = os.getenv("RVEDA_DB_PATH")
    if env_path:
        return Path(env_path)

    codex_memory_root = Path.home() / ".codex" / "memories"
    if codex_memory_root.exists():
        return codex_memory_root / "rveda" / "icd10.db"

    return ROOT_DIR / "data" / "icd10.db"


DB_PATH = _default_db_path()


def _conn_has_rows(conn: sqlite3.Connection) -> bool:
    """Return True when the current SQLite connection has populated ICD-10 rows."""
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type = 'table' AND name = 'icd10_codes'
        """
    ).fetchone()
    if not row or row[0] == 0:
        return False

    count = conn.execute("SELECT COUNT(*) FROM icd10_codes").fetchone()
    return bool(count and count[0] > 0)


def _db_has_rows(db_path: Path) -> bool:
    """Return True when the packaged SQLite database already has ICD-10 rows."""
    if not db_path.exists():
        return False

    try:
        with sqlite3.connect(db_path) as conn:
            return _conn_has_rows(conn)
    except sqlite3.Error:
        return False


def initialize_db(
    db_path: Path = DB_PATH,
    data_path: Path = DATA_PATH,
) -> None:
    """Create the local SQLite database and load mock ICD-10 records if needed."""
    if _db_has_rows(db_path):
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        # Serialize first-write bootstrap to avoid concurrent DELETE/INSERT races.
        conn.execute("BEGIN IMMEDIATE")
        if _conn_has_rows(conn):
            conn.commit()
            return

        with data_path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS icd10_codes (
                code TEXT PRIMARY KEY,
                short_desc TEXT NOT NULL,
                long_desc TEXT NOT NULL,
                excludes TEXT NOT NULL
            )
            """
        )
        conn.execute("DELETE FROM icd10_codes")
        conn.executemany(
            """
            INSERT INTO icd10_codes (code, short_desc, long_desc, excludes)
            VALUES (:code, :short_desc, :long_desc, :excludes)
            """,
            records,
        )
        conn.commit()


def search_codes(query: str, limit: int = 5) -> list[dict]:
    """Search ICD-10 records by description and return code summaries."""
    pattern = f"%{query}%"
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT code, short_desc
            FROM icd10_codes
            WHERE short_desc LIKE ? OR long_desc LIKE ?
            ORDER BY code
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def get_code_details(code: str) -> dict:
    """Return long description and exclusion notes for an exact ICD-10 code."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT long_desc, excludes
            FROM icd10_codes
            WHERE code = ?
            """,
            (code,),
        ).fetchone()

    return dict(row) if row is not None else {}


if __name__ == "__main__":
    initialize_db()
    print(search_codes("diabetes"))
