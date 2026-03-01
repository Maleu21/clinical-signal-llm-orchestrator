from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .model import ECGConvNet
from .orchestrator import SignalResult


def load_model(ckpt_path: str = "models/ecg_cnn.pt") -> ECGConvNet | None:
    p = Path(ckpt_path)
    if not p.exists():
        return None
    model = ECGConvNet()
    ckpt = torch.load(p, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def infer_signal(ecg_window: np.ndarray, model: ECGConvNet | None = None) -> SignalResult:
    """
    ecg_window: shape (L,)
    """
    if model is None:
        # fallback heuristic if no trained model yet
        v = float(np.var(ecg_window))
        risk = min(1.0, max(0.0, (v - 0.5) / 2.0))
        label = "likely abnormal" if risk > 0.6 else "likely normal"
        note = "fallback heuristic (train CNN to replace)."
        return SignalResult(risk_score=risk, label=label, note=note)

    x = torch.from_numpy(ecg_window.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    with torch.no_grad():
        logits = model(x)
        prob_abnormal = float(F.softmax(logits, dim=1)[0, 1].item())
    label = "likely abnormal" if prob_abnormal >= 0.5 else "likely normal"
    note = "CNN inference on MIT-BIH windows."
    return SignalResult(risk_score=prob_abnormal, label=label, note=note)