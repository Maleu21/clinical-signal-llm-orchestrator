import os
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/signals")


# =========================
# Connexion
# =========================
def get_connection():
    return psycopg2.connect(DATABASE_URL)


# =========================
# Initialisation des tables
# =========================
def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ecg_windows (
            id SERIAL PRIMARY KEY,
            record_id TEXT,
            label INTEGER,
            signal BYTEA
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            risk_score REAL,
            label INTEGER
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


# =========================
# Sauvegarde d'une prédiction
# =========================
def save_prediction(risk_score: float, label: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO predictions (timestamp, risk_score, label)
        VALUES (%s, %s, %s)
        """,
        (datetime.now(timezone.utc).isoformat(), risk_score, label),
    )

    conn.commit()
    cur.close()
    conn.close()


# =========================
# Historique des prédictions
# =========================
def get_recent_predictions(limit: int = 10) -> list:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute(
        """
        SELECT id, timestamp, risk_score, label
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT %s
        """,
        (limit,),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(row) for row in rows]