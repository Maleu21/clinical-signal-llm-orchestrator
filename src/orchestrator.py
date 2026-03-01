from dataclasses import dataclass
from .prompts import SYSTEM_PROMPT, DECISION_PROMPT, RESPONSE_PROMPT
from .evals import check_format, check_safety

@dataclass
class SignalResult:
    risk_score: float  # 0..1
    label: str         # "likely normal" / "likely abnormal"
    note: str          # short explanation

def summarize_signal(sr: SignalResult) -> str:
    return f"risk_score={sr.risk_score:.2f}, label={sr.label}, note={sr.note}"

def decide_next_step(sr: SignalResult) -> str:
    # Rule-based decision for demo (replace by LLM later)
    if sr.risk_score >= 0.75:
        return "ESCALATE"
    if sr.risk_score >= 0.45:
        return "CLARIFY"
    return "REASSURE"

def llm_generate_response(patient_utterance: str, signal_summary: str, next_step: str) -> str:
    # Demo mode: mock LLM response. Later: call OpenAI/other provider.
    if next_step == "ESCALATE":
        return ("Je comprends. Compte tenu des éléments, je vous conseille de contacter "
                "un professionnel de santé rapidement ou un service d’urgence si les symptômes "
                "s’aggravent. Êtes-vous seul(e) actuellement ?")
    if next_step == "CLARIFY":
        return ("Merci. Pour mieux comprendre : depuis quand ressentez-vous ces palpitations, "
                "et avez-vous eu un malaise ou une douleur thoracique ?")
    return ("Merci pour ces informations. À ce stade, rien n’indique une urgence immédiate, "
            "mais si les symptômes persistent ou s’aggravent, contactez un professionnel de santé. "
            "Souhaitez-vous que je planifie un rendez-vous ?")

class CallOrchestrator:
    def run(self, patient_utterance: str, signal_result: SignalResult) -> dict:
        signal_summary = summarize_signal(signal_result)
        next_step = decide_next_step(signal_result)

        response = llm_generate_response(patient_utterance, signal_summary, next_step)

        ok = check_format(response) and check_safety(response)
        if not ok:
            response = ("Je ne peux pas vous donner de diagnostic. "
                        "Si vous êtes inquiet(e), contactez un professionnel de santé.")

        return {
            "patient_utterance": patient_utterance,
            "signal_summary": signal_summary,
            "next_step": next_step,
            "agent_response": response,
        }