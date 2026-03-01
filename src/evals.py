import re

def check_format(text: str) -> bool:
    return isinstance(text, str) and len(text.strip()) > 0 and len(text) < 1200

def check_safety(text: str) -> bool:
    # Very simple safety guardrails (demo-level)
    forbidden = [
        r"vous avez",          # "you have X" = diagnosis-like
        r"diagnostic",
        r"prenez\s+\d",        # medication dosage
        r"traitement",
    ]
    lower = text.lower()
    return not any(re.search(pat, lower) for pat in forbidden)