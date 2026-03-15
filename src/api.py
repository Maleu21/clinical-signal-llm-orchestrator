from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

from src.model import ECGConvNet
from src.db import init_db, save_prediction

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
except Exception as e:
    print("⚠️ Model loading failed:", e)

model.to(device)
model.eval()

# =========================
# Predict endpoint
# =========================

@app.post("/predict")
def predict(request: PredictRequest):

    window = request.window

    # Vérification taille
    if len(window) != 720:
        raise HTTPException(
            status_code=400,
            detail="Window must be 720 samples (2 seconds at 360Hz)"
        )

    # Conversion en tensor
    x = np.array(window, dtype=np.float32)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    label = int(prob > 0.5)

    # Sauvegarde en base
    save_prediction(prob, label)

# Retour de la réponse
    return {
        "risk_score": float(prob),
        "label": label
    }