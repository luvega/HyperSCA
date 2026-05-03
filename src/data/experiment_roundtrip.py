"""Wet-lab <-> model roundtrip helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validators import validate_experiment_roundtrip_fields


def load_experiment_results(path: str | Path) -> pd.DataFrame:
    """Load experiment roundtrip table from csv/json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Experiment file not found: {p}")
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        df = pd.DataFrame(obj)
    else:
        df = pd.read_csv(p)
    issues = validate_experiment_roundtrip_fields(df)
    if issues:
        raise ValueError("Invalid experiment table: " + "; ".join(issues))
    return df


def summarize_experiment_effects(df: pd.DataFrame) -> dict:
    """Aggregate experiment effects by gene and dose."""
    out = {}
    for gene, sub in df.groupby("gene"):
        by_dose = sub.groupby("dose")["effect_size"].mean()
        out[str(gene)] = {
            "n_rows": int(len(sub)),
            "mean_effect": float(sub["effect_size"].mean()),
            "std_effect": float(sub["effect_size"].std(ddof=0)),
            "dose_effect": {str(k): float(v) for k, v in by_dose.items()},
        }
    return out


def calibrate_pkpd_params(
    effect_summary: dict,
    *,
    default_ec50: float = 1.0,
    default_emax: float = 1.0,
) -> dict:
    """Simple PK/PD parameter calibration from experiment summary."""
    genes = list(effect_summary.keys())
    if not genes:
        return {"ec50": default_ec50, "emax": default_emax}

    max_effects = []
    ec50_proxy = []
    for g in genes:
        s = effect_summary[g]
        max_effects.append(abs(float(s.get("mean_effect", 0.0))))
        doses = s.get("dose_effect", {})
        if doses:
            # pick dose nearest to half max effect as EC50 proxy
            vals = [(float(d), abs(float(e))) for d, e in doses.items()]
            vals.sort(key=lambda x: x[0])
            target = 0.5 * max(v for _, v in vals)
            best = min(vals, key=lambda x: abs(x[1] - target))
            ec50_proxy.append(best[0])

    emax = float(np.clip(np.mean(max_effects), 0.1, 2.0)) if max_effects else default_emax
    ec50 = float(np.clip(np.mean(ec50_proxy), 0.05, 20.0)) if ec50_proxy else default_ec50
    return {"ec50": ec50, "emax": emax}
