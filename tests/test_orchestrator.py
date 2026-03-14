from src.orchestrator import CallOrchestrator, SignalResult


def test_orchestrator_response_structure():

    orch = CallOrchestrator()

    sr = SignalResult(
        risk_score=0.9,
        label="likely abnormal",
        note="test"
    )

    result = orch.run(
        patient_utterance="Je me sens étourdi et j'ai des palpitations.",
        signal_result=sr
    )

    assert "next_step" in result
    assert "signal_summary" in result
    assert "agent_response" in result