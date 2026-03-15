from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import time

from src.model import ECGConvNet
from src.db import init_db, save_prediction, get_recent_predictions
from src.validation import validate_signal
from src.logger import get_logger

logger = get_logger(__name__)

# =========================
# App initialization
# =========================

app = FastAPI(title="ECG Risk API")

device = torch.device("cpu")

# Initialiser la base au démarrage
init_db()

# =========================
# Request schema
# =========================

class PredictRequest(BaseModel):
    window: list[float]

# =========================
# Load model
# =========================

model = ECGConvNet()

try:
    checkpoint = torch.load("models/ecg_cnn.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    logger.info("Modele charge avec succes")
except Exception as e:
    logger.warning("Modele non trouve, poids aleatoires utilises", extra={"error": str(e)})

model.to(device)
model.eval()

# =========================
# Predict endpoint
# =========================

@app.post("/predict")
def predict(request: PredictRequest):
    start = time.time()

    # Validation du signal
    x = validate_signal(request.window)

    # Conversion en tensor
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    label = int(prob > 0.5)
    duration_ms = round((time.time() - start) * 1000, 2)

    # Sauvegarde en base
    save_prediction(prob, label)

    logger.info("Prediction effectuee", extra={
        "risk_score": round(prob, 4),
        "label": label,
        "duration_ms": duration_ms,
    })

    return {
        "risk_score": float(prob),
        "label": label
    }


# =========================
# History endpoint
# =========================

@app.get("/history")
def history(limit: int = 10):
    rows = get_recent_predictions(limit=limit)
    return {"predictions": rows}


# =========================
# Health endpoint
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}