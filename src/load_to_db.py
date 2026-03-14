import numpy as np
import pickle

from src.db import init_db, get_connection
from src.data import load_npz


def main():

    print("Loading dataset")

    d = load_npz("data/processed/mitbih_windows.npz")

    init_db()

    conn = get_connection()
    cur = conn.cursor()

    for i in range(len(d.X)):

        signal = pickle.dumps(d.X[i])

        cur.execute(
            """
            INSERT INTO ecg_windows (record_id, label, signal)
            VALUES (?, ?, ?)
            """,
            (str(d.record_id[i]), int(d.y[i]), signal),
        )

        if i % 2000 == 0:
            print(f"Inserted {i}")

    conn.commit()
    conn.close()

    print("Database ready")


if __name__ == "__main__":
    main()