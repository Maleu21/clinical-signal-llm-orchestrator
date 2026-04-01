import os
import requests
from dataclasses import dataclass
from .prompts import SYSTEM_PROMPT, DECISION_PROMPT, RESPONSE_PROMPT
from .evals import check_format, check_safety


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


@dataclass
class SignalResult:
    risk_score: float  # 0..1
    label: str         # "likely normal" / "likely abnormal"
    note: str          # short explanation


def summarize_signal(sr: SignalResult) -> str:
    return f"risk_score={sr.risk_score:.2f}, label={sr.label}, note={sr.note}"


def _call_ollama(prompt: str, system: str, timeout: int = 120) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ]
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return None


def decide_next_step(sr: SignalResult, patient_utterance: str) -> str:
    """Décision via LLM — fallback rule-based si Ollama indisponible."""
    signal_summary = summarize_signal(sr)

    prompt = DECISION_PROMPT.format(
        patient_utterance=patient_utterance,
        signal_summary=signal_summary,
    )

    result = _call_ollama(prompt, SYSTEM_PROMPT)

    # Vérification que la réponse est valide
    if result and result.upper() in ("ESCALATE", "CLARIFY", "REASSURE"):
        return result.upper()

    # Fallback rule-based si LLM indisponible ou réponse invalide
    print("⚠️ Fallback to rule-based decision")
    if sr.risk_score >= 0.75:
        return "ESCALATE"
    if sr.risk_score >= 0.45:
        return "CLARIFY"
    return "REASSURE"


def llm_generate_response(patient_utterance: str, signal_summary: str, next_step: str) -> str:
    """Génère la réponse de l'agent via LLM — fallback si Ollama indisponible."""
    prompt = RESPONSE_PROMPT.format(
        patient_utterance=patient_utterance,
        signal_summary=signal_summary,
        next_step=next_step,
    )

    result = _call_ollama(prompt, SYSTEM_PROMPT)

    if result:
        # Guardrails de sécurité sur la réponse LLM
        if not check_format(result) or not check_safety(result):
            print("⚠️ Réponse LLM bloquée par les guardrails")
            return ("Je ne peux pas vous donner de diagnostic. "
                    "Si vous êtes inquiet(e), contactez un professionnel de santé.")
        return result

    # Fallback mock si Ollama indisponible
    print("⚠️ Fallback to mock response")
    if next_step == "ESCALATE":
        return ("Je comprends. Compte tenu des éléments, je vous conseille de contacter "
                "un professionnel de santé rapidement ou un service d'urgence si les symptômes "
                "s'aggravent. Êtes-vous seul(e) actuellement ?")
    if next_step == "CLARIFY":
        return ("Merci. Pour mieux comprendre : depuis quand ressentez-vous ces palpitations, "
                "et avez-vous eu un malaise ou une douleur thoracique ?")
    return ("Merci pour ces informations. À ce stade, rien n'indique une urgence immédiate, "
            "mais si les symptômes persistent ou s'aggravent, contactez un professionnel de santé. "
            "Souhaitez-vous que je planifie un rendez-vous ?")


class CallOrchestrator:
    def run(self, patient_utterance: str, signal_result: SignalResult) -> dict:
        signal_summary = summarize_signal(signal_result)
        next_step = decide_next_step(signal_result, patient_utterance)
        response = llm_generate_response(patient_utterance, signal_summary, next_step)

        return {
            "patient_utterance": patient_utterance,
            "signal_summary": signal_summary,
            "next_step": next_step,
            "agent_response": response,
        }