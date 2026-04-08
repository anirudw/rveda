"""SQLite-backed ICD-10 search utilities for the Rveda environment."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "icd10.db"
DATA_PATH = ROOT_DIR / "icd10_mock.json"


def initialize_db(
    db_path: Path = DB_PATH,
    data_path: Path = DATA_PATH,
) -> None:
    """Create the local SQLite database and load mock ICD-10 records."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    with sqlite3.connect(db_path) as conn:
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
