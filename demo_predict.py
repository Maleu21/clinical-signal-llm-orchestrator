import requests
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"


# ==========================================================
# 1️⃣ TEST 1 — Signal zéro
# ==========================================================
def test_zero_signal():
    print("\n===== TEST 1 — Signal ZÉRO =====")
    signal = np.zeros(720).tolist()

    send_request(signal)


# ==========================================================
# 2️⃣ TEST 2 — Signal avec petit bruit
# ==========================================================
def test_small_noise():
    print("\n===== TEST 2 — Petit BRUIT =====")

    # signal de base très faible
    base = np.zeros(720)

    # ajout bruit léger
    noise = np.random.normal(0, 0.05, 720)

    signal = (base + noise).tolist()

    send_request(signal)


# ==========================================================
# 3️⃣ TEST 3 — Signal avec spike fort
# ==========================================================
def test_spike():
    print("\n===== TEST 3 — SPIKE FORT =====")

    t = np.linspace(0, 2, 720)

    # signal sinusoïdal
    signal = 0.5 * np.sin(2 * np.pi * 5 * t)

    # ajouter un spike fort au milieu
    spike_position = 360
    signal[spike_position:spike_position+5] += 2.0

    signal = signal.tolist()

    send_request(signal)


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
    test_small_noise()
    test_spike()
