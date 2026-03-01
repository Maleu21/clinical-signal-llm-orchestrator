import numpy as np

from src.data import load_npz
from src.infer import load_model, infer_signal
from src.orchestrator import CallOrchestrator


def main():
    patient_utterance = "J’ai des palpitations et je me sens un peu étourdi depuis ce matin."

    d = load_npz("data/processed/mitbih_windows.npz")

    # Pick a random sample (for demo)
    rng = np.random.default_rng(0)
    idx = int(rng.integers(0, len(d.X)))
    ecg_window = d.X[idx]
    true_label = int(d.y[idx])

    model = load_model("models/ecg_cnn.pt")
    sr = infer_signal(ecg_window, model=model)

    orch = CallOrchestrator()
    result = orch.run(patient_utterance, sr)

    print("\n=== DEMO OUTPUT (REAL DATA) ===")
    print(f"Sample idx={idx} true_label={true_label} (0=normal,1=abnormal)")
    print("Next step:", result["next_step"])
    print("Signal:", result["signal_summary"])
    print("Agent:", result["agent_response"])


if __name__ == "__main__":
    main()