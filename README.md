# Clinical Signal LLM Orchestrator

Système de télésurveillance médicale simulant la fusion d'un signal ECG connecté et d'une utterance patient pour aider un agent téléphonique à prendre de meilleures décisions cliniques.

En production, le signal ECG proviendrait d'un appareil connecté (montre, patch ECG) et l'utterance d'une transcription audio en temps réel. Dans cette démo, les deux entrées sont simulées pour illustrer l'architecture complète.

## Architecture
```
Appareil ECG connecté          Patient (téléphone)
        ↓                               ↓
Signal ECG (720 samples)       Utterance (texte simulé)
        ↓                               ↓
Validation (NaN, plat,                  |
amplitude, taille)                      |
        ↓                               |
ECGConvNet (CNN 1D)                     |
→ risk_score (0..1)                     |
        ↓                               ↓
        └──────── CallOrchestrator ─────┘
                        ↓
              ESCALATE / CLARIFY / REASSURE
                        ↓
              Mistral 7B (Ollama)
                        ↓
              Réponse en français pour l'agent
```

## Stack technique

- **FastAPI** — API REST
- **PyTorch** — CNN 1D pour classification ECG
- **Ollama + Mistral 7B** — LLM local, pas de clé API requise
- **PostgreSQL** — persistance des prédictions
- **Docker Compose** — orchestration des services

## Prérequis

- Docker + Docker Compose
- 8 Go de RAM minimum
- GPU NVIDIA recommandé (fonctionne aussi sur CPU)

## Lancement
```bash
git clone https://github.com/Maleu21/clinical-signal-llm-orchestrator.git
cd clinical-signal-llm-orchestrator
chmod +x start.sh
./start.sh
```

Le script fait tout automatiquement :
1. Lance l'API FastAPI + PostgreSQL + Ollama via Docker Compose
2. Télécharge Mistral 7B (4.1 Go, **une seule fois**)
3. Télécharge et prépare les données MIT-BIH (**une seule fois**)
4. Entraîne le CNN sur les données ECG (**une seule fois**, ~5 min)

L'API est disponible sur `http://localhost:8000`
La documentation interactive sur `http://localhost:8000/docs`

## Démo

Une fois le projet lancé :
```bash
# Pipeline complète : signal ECG réel → CNN → LLM → réponse agent
docker-compose exec ecg-api python3 demo.py

# Tester les cas de validation (signal plat, NaN, amplitude...)
docker-compose exec ecg-api python3 demo_predict.py

# Tests unitaires
docker-compose exec ecg-api pytest tests/ -v
```

## Validation du signal

L'API rejette automatiquement les signaux invalides :

| Cas | Code | Message |
|-----|------|---------|
| Taille incorrecte | 422 | Taille invalide : X samples reçus, 720 attendus |
| Valeurs NaN | 422 | Signal invalide : contient des valeurs NaN |
| Signal plat | 422 | Signal invalide : variance trop faible |
| Amplitude hors normes | 422 | Signal invalide : amplitude hors normes |

## Structure du projet
```
src/
├── api.py          # Endpoint FastAPI
├── model.py        # Architecture CNN 1D
├── orchestrator.py # Logique de décision + appel LLM
├── prompts.py      # Prompts DECISION et RESPONSE
├── validation.py   # Validation des signaux ECG
├── logger.py       # Logs JSON structurés
├── evals.py        # Guardrails de sécurité médicale
├── data.py         # Dataset PyTorch
├── train.py        # Entraînement du modèle
└── db.py           # Persistance PostgreSQL
tests/
├── test_validation.py
├── test_orchestrator.py
└── test_evals.py
```

## Décisions de conception

**Pourquoi Ollama ?** LLM local, pas de dépendance à une API externe, fonctionne hors ligne, gratuit.

**Pourquoi un fallback rule-based ?** Si Ollama est indisponible, l'orchestrateur bascule automatiquement sur des règles déterministes - le système reste opérationnel.

**Guardrails médicaux** Toute réponse LLM est filtrée par `check_safety()` pour éviter tout diagnostic ou conseil de traitement.

**Limites et évolutions possibles**
- L'utterance patient est simulée - en production elle viendrait d'une transcription audio temps réel (Whisper)
- Le signal ECG est simulé - en production il viendrait d'un appareil connecté via API
- Airflow pourrait orchestrer le réentraînement automatique du CNN
- Prometheus + Grafana pour monitorer les drifts de distribution en production