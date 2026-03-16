import requests
import numpy as np
import sys
sys.path.insert(0, '/app')

from src.data import load_npz
from src.infer import load_model, infer_signal

API_URL = "http://localhost:8000/predict"


# ==========================================================
# 1 — Signal zéro (rejeté par validation)
# ==========================================================
def test_zero_signal():
    print("\n===== TEST 1 — Signal ZERO =====")
    signal = np.zeros(720).tolist()
    send_request(signal)


# ==========================================================
# 2 — Vrai signal ECG normal
# ==========================================================
def test_normal_signal():
    print("\n===== TEST 2 — Signal ECG NORMAL =====")
    d = load_npz("data/processed/mitbih_windows.npz")
    normal_idx = next(i for i, y in enumerate(d.y) if y == 0)
    signal = d.X[normal_idx].tolist()
    send_request(signal)


# ==========================================================
# 3 — Vrai signal ECG anormal bien détecté
# ==========================================================
def test_abnormal_signal():
    print("\n===== TEST 3 — Signal ECG ANORMAL =====")
    d = load_npz("data/processed/mitbih_windows.npz")
    model = load_model("models/ecg_cnn.pt")
    for i, y in enumerate(d.y):
        if y == 1:
            sr = infer_signal(d.X[i], model=model)
            if sr.risk_score > 0.7:
                print(f"Sample idx={i}")
                send_request(d.X[i].tolist())
                break


# ==========================================================
# Fonction d'appel API
# ==========================================================
def send_request(signal):
    response = requests.post(
        API_URL,
        json={"window": signal}
    )
    if response.status_code == 200:
        result = response.json()
        print("Risk score:", round(result["risk_score"], 4))
        print("Label:", result["label"])
    else:
        print("Error:", response.status_code, response.text)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    test_zero_signal()
    test_normal_signal()
    test_abnormal_signal()