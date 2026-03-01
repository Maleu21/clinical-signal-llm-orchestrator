import numpy as np
from .orchestrator import SignalResult

def infer_signal(ecg_window: np.ndarray) -> SignalResult:
    # Demo heuristic: higher variance => higher "risk"
    v = float(np.var(ecg_window))
    risk = min(1.0, max(0.0, (v - 0.5) / 2.0))  # crude scaling
    label = "likely abnormal" if risk > 0.6 else "likely normal"
    note = "demo heuristic based on signal variance (replace with CNN)."
    return SignalResult(risk_score=risk, label=label, note=note)

def make_synthetic_ecg(length: int = 2000, noise: float = 0.15, abnormal: bool = False) -> np.ndarray:
    t = np.linspace(0, 6.0, length)
    base = np.sin(2 * np.pi * 1.2 * t)
    if abnormal:
        base += 0.6 * np.sin(2 * np.pi * 3.5 * t)  # extra component
        noise *= 2.0
    return base + noise * np.random.randn(length)