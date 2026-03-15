import numpy as np
from fastapi import HTTPException


# Seuils physiologiques ECG (signal MIT-BIH normalisé)
MIN_VARIANCE = 1e-4       # signal plat en dessous
MAX_AMPLITUDE = 10.0      # saturation au dessus
MIN_AMPLITUDE = -10.0     # saturation négative


def validate_signal(window: list[float]) -> np.ndarray:
    """
    Valide un signal ECG brut avant inférence.
    Lève une HTTPException 422 si le signal est invalide.
    """

    # 1. Taille
    if len(window) != 720:
        raise HTTPException(
            status_code=422,
            detail=f"Taille invalide : {len(window)} samples reçus, 720 attendus (2s à 360Hz)."
        )

    x = np.array(window, dtype=np.float32)

    # 2. Valeurs nulles / NaN / infinies
    if np.any(np.isnan(x)):
        raise HTTPException(
            status_code=422,
            detail="Signal invalide : contient des valeurs NaN."
        )
    if np.any(np.isinf(x)):
        raise HTTPException(
            status_code=422,
            detail="Signal invalide : contient des valeurs infinies."
        )

    # 3. Signal plat (variance trop faible)
    if np.var(x) < MIN_VARIANCE:
        raise HTTPException(
            status_code=422,
            detail=f"Signal invalide : variance trop faible ({np.var(x):.2e}). Signal plat ou déconnecté."
        )

    # 4. Amplitude hors normes (saturation)
    if np.any(x > MAX_AMPLITUDE) or np.any(x < MIN_AMPLITUDE):
        raise HTTPException(
            status_code=422,
            detail=f"Signal invalide : amplitude hors normes (min={x.min():.2f}, max={x.max():.2f})."
        )

    return x