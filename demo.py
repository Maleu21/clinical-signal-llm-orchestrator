from src.infer import make_synthetic_ecg, infer_signal
from src.orchestrator import CallOrchestrator

def main():
    patient_utterance = "J’ai des palpitations et je me sens un peu étourdi depuis ce matin."
    ecg = make_synthetic_ecg(abnormal=True)
    sr = infer_signal(ecg)

    orch = CallOrchestrator()
    result = orch.run(patient_utterance, sr)

    print("\n=== DEMO OUTPUT ===")
    print("Next step:", result["next_step"])
    print("Signal:", result["signal_summary"])
    print("Agent:", result["agent_response"])

if __name__ == "__main__":
    main()