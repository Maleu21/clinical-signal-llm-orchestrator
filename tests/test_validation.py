import pytest
import numpy as np
from fastapi import HTTPException
from src.validation import validate_signal


def make_valid_signal():
    """Signal ECG valide de référence."""
    np.random.seed(42)
    return list(np.random.randn(720).astype(np.float32))


# =========================
# Cas valides
# =========================

def test_valid_signal_returns_array():
    signal = make_valid_signal()
    result = validate_signal(signal)
    assert isinstance(result, np.ndarray)
    assert result.shape == (720,)


# =========================
# Taille incorrecte
# =========================

def test_too_short_raises():
    signal = make_valid_signal()[:500]
    with pytest.raises(HTTPException) as exc:
        validate_signal(signal)
    assert exc.value.status_code == 422
    assert "500" in exc.value.detail


def test_too_long_raises():
    signal = make_valid_signal() + [0.1] * 100
    with pytest.raises(HTTPException) as exc:
        validate_signal(signal)
    assert exc.value.status_code == 422


def test_empty_raises():
    with pytest.raises(HTTPException) as exc:
        validate_signal([])
    assert exc.value.status_code == 422


# =========================
# NaN / infinies
# =========================

def test_nan_raises():
    signal = make_valid_signal()
    signal[100] = float("nan")
    with pytest.raises(HTTPException) as exc:
        validate_signal(signal)
    assert "NaN" in exc.value.detail


def test_inf_raises():
    signal = make_valid_signal()
    signal[200] = float("inf")
    with pytest.raises(HTTPException) as exc:
        validate_signal(signal)
    assert "infinie" in exc.value.detail


def test_neg_inf_raises():
    signal = make_valid_signal()
    signal[300] = float("-inf")
    with pytest.raises(HTTPException) as exc:
        validate_signal(signal)
    assert "infinie" in exc.value.detail


# =========================
# Signal plat
# =========================

def test_flat_signal_raises():
    signal = [0.0] * 720
    with pytest.raises(HTTPException) as exc:
        validate_signal(signal)
    assert "plat" in exc.value.detail


def test_near_flat_signal_raises():
    signal = [1e-6] * 720
    with pytest.raises(HTTPException) as exc:
        validate_signal(signal)
    assert exc.value.status_code == 422


# =========================
# Amplitude hors normes
# =========================

def test_amplitude_too_high_raises():
    signal = make_valid_signal()
    signal[50] = 15.0
    with pytest.raises(HTTPException) as exc:
        validate_signal(signal)
    assert "amplitude" in exc.value.detail


def test_amplitude_too_low_raises():
    signal = make_valid_signal()
    signal[50] = -15.0
    with pytest.raises(HTTPException) as exc:
        validate_signal(signal)
    assert "amplitude" in exc.value.detail


def test_amplitude_at_limit_is_valid():
    signal = make_valid_signal()
    # Exactement à la limite — doit passer
    signal[50] = 10.0
    result = validate_signal(signal)
    assert result is not None