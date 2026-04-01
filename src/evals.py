import re

def check_format(text: str) -> bool:
    return isinstance(text, str) and len(text.strip()) > 0 and len(text) < 1200

def check_safety(text: str) -> bool:
    # Guardrails médicaux — éviter tout diagnostic ou prescription
    forbidden = [
        r"vous avez",
        r"\b(arythmie|infarctus|fibrillation|tachycardie|bradycardie|cancer)\b",
        r"diagnostic(quer|qué)?",
        r"prenez\s+\d",
        r"posologie",
        r"ordonnance",
    ]
    lower = text.lower()
    return not any(re.search(pat, lower) for pat in forbidden)