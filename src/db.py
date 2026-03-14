import sqlite3
from pathlib import Path

DB_PATH = Path("data/ecg.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ecg_windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT,
            label INTEGER,
            signal BLOB
        )
        """
    )

    conn.commit()
    conn.close()