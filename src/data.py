from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NpzData:
    X: np.ndarray          # (N, L) float32
    y: np.ndarray          # (N,) int64
    record_id: np.ndarray  # (N,) object/str
    fs: int
    window_size: int


def load_npz(path: str) -> NpzData:
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    record_id = d["record_id"]
    fs = int(d["fs"])
    window_size = int(d["window_size"])
    return NpzData(X=X, y=y, record_id=record_id, fs=fs, window_size=window_size)


class ECGWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        # shape to (C, L) for Conv1D
        x = torch.from_numpy(x).unsqueeze(0)  # (1, L)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y