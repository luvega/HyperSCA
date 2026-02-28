"""Dose-response functions for dynamic intervention."""
from __future__ import annotations

import numpy as np


def hill_effect(
    concentration: np.ndarray,
    *,
    emax: float = 1.0,
    ec50: float = 1.0,
    hill: float = 1.2,
) -> np.ndarray:
    """Hill equation effect in [0, emax]."""
    c = np.asarray(concentration, dtype=float)
    num = np.power(c, hill)
    den = np.power(ec50, hill) + num + 1e-12
    return np.clip(emax * num / den, 0.0, emax)


def summarize_dose_response(
    pk_curves: dict[float, np.ndarray],
    *,
    emax: float,
    ec50: float,
    hill: float,
) -> dict[float, dict]:
    """Summarize per-dose pharmacodynamic strength."""
    out: dict[float, dict] = {}
    for dose, conc in pk_curves.items():
        effect = hill_effect(conc, emax=emax, ec50=ec50, hill=hill)
        out[float(dose)] = {
            "auc_concentration": float(np.trapezoid(conc)),
            "auc_effect": float(np.trapezoid(effect)),
            "max_effect": float(np.max(effect)),
            "mean_effect": float(np.mean(effect)),
            "effect_curve": effect.tolist(),
        }
    return out
