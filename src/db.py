import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# Base de données locale
DB_PATH = Path("data/ecg.db")


# =========================
# Connexion
# =========================
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


# =========================
# Initialisation des tables
# =========================
def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Table des signaux (optionnel si tu veux stocker les données)
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

    # Table des prédictions (IMPORTANT pour ton API)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            risk_score REAL,
            label INTEGER
        )
        """
    )

    conn.commit()
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
        VALUES (?, ?, ?)
        """,
        (datetime.now(timezone.utc).isoformat(), risk_score, label),
    )

    conn.commit()
    conn.close()