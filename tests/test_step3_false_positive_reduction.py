"""Tests for Step3 false-positive filtering."""
from __future__ import annotations

import pandas as pd

from src.perturbation.false_positive_filter import filter_false_positive_targets


def test_false_positive_filter_reduces_candidates():
    df = pd.DataFrame(
        {
            "ligand": ["A", "B", "C", "D"],
            "receptor": ["R1", "R2", "R3", "R4"],
            "target_priority_score": [0.5, 0.2, 0.01, 0.005],
            "combined_abs_delta": [0.9, 0.4, 0.05, 0.01],
            "prior_hit": [True, False, False, False],
        }
    )
    out = filter_false_positive_targets(df, min_score=0.02, max_fdr=0.5)
    assert "pass_filter" in out.columns
    kept = out[out["pass_filter"]]
    assert len(kept) < len(df)
    assert len(kept) >= 1
