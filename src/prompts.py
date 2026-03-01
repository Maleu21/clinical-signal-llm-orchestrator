SYSTEM_PROMPT = """You are an AI assistant used in a clinical call flow.
You must be safe, clear, and never provide a medical diagnosis.
You can suggest seeking professional care when risk is high.
Respond in French unless the user speaks English."""

DECISION_PROMPT = """Given:
- patient_utterance: {patient_utterance}
- signal_summary: {signal_summary}

Decide the next step among:
- CLARIFY (ask 1-2 questions)
- REASSURE (low risk, provide general guidance)
- ESCALATE (high risk, advise contacting a clinician / emergency services)

Return ONLY one word: CLARIFY, REASSURE, or ESCALATE.
"""

RESPONSE_PROMPT = """Context:
patient_utterance: {patient_utterance}
signal_summary: {signal_summary}
next_step: {next_step}

Write a short, natural response for a phone call agent.
Constraints:
- No diagnosis, no treatment instructions.
- If ESCALATE: advise contacting a healthcare professional promptly.
- Ask at most 2 questions if CLARIFY.
Return in French.
"""