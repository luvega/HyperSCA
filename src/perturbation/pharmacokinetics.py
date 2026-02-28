"""PK models for dynamic intervention simulation."""
from __future__ import annotations

import numpy as np


def one_compartment_oral(
    dose: float,
    time_grid: np.ndarray,
    ka: float = 1.2,
    ke: float = 0.18,
    vd: float = 1.0,
    bioavailability: float = 1.0,
) -> np.ndarray:
    """One-compartment oral PK concentration curve.

    C(t) = F * Dose * ka / (Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
    """
    t = np.asarray(time_grid, dtype=float)
    denom = max(vd * (ka - ke), 1e-8)
    conc = bioavailability * dose * ka / denom * (np.exp(-ke * t) - np.exp(-ka * t))
    return np.clip(conc, 0.0, None)


def simulate_pk_grid(
    dose_grid: list[float],
    time_grid: list[float],
    *,
    ka: float,
    ke: float,
    vd: float,
    bioavailability: float,
) -> dict[float, np.ndarray]:
    """Simulate concentration-time curves for all doses."""
    t = np.asarray(time_grid, dtype=float)
    return {
        float(d): one_compartment_oral(
            float(d),
            t,
            ka=ka,
            ke=ke,
            vd=vd,
            bioavailability=bioavailability,
        )
        for d in dose_grid
    }
