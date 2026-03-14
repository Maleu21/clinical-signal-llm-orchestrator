from __future__ import annotations

import os
from signal import signal
import numpy as np
import wfdb
from pathlib import Path


# =========================
# Configuration
# =========================

RECORDS = [
    "100", "101", "102", "103", "104",
    "105", "106", "107", "108", "109"
]

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

WINDOW_SECONDS = 2.0
FS = 360  # MIT-BIH sampling rate


# =========================
# Data Download
# =========================

def download_record(record: str):
    record_path = RAW_DIR / record

    if not record_path.exists():
        print(f"[prepare] Downloading record {record}")
        wfdb.dl_database("mitdb", dl_dir=str(RAW_DIR))


# =========================
# Windowing
# =========================

def create_windows(signal, annotations, window_size):
    X, y = [], []

    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i + window_size]
        label = 0  # default normal

        # if any abnormal beat inside window → label 1
        for idx, ann_symbol in zip(annotations.sample, annotations.symbol):
            if i <= idx < i + window_size:
                if ann_symbol != "N":
                    label = 1
                    break

        X.append(window)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# =========================
# Main
# =========================

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_X = []
    all_y = []
    all_record_id = []

    window_size = int(WINDOW_SECONDS * FS)

    for record in RECORDS:
        download_record(record)

        record_path = RAW_DIR / record
        signal, fields = wfdb.rdsamp(str(record_path))
        assert signal is not None

        ann = wfdb.rdann(str(record_path), "atr")

        X, y = create_windows(signal[:, 0], ann, window_size) 

        all_X.append(X)
        all_y.append(y)
        all_record_id.extend([record] * len(y))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    record_id = np.array(all_record_id)

    output_path = PROCESSED_DIR / "mitbih_windows.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        record_id=record_id,
        fs=FS,
        window_size=window_size,
    )

    print(f"[prepare] Saved to {output_path}")
    print(f"[prepare] Final shape: {X.shape}")


if __name__ == "__main__":
    main()