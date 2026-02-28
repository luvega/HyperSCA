"""Combinatorial intervention ranking."""
from __future__ import annotations

from itertools import combinations


def generate_target_combinations(
    targets: list[str],
    max_size: int = 2,
) -> list[tuple[str, ...]]:
    """Generate target combinations up to max_size."""
    combos: list[tuple[str, ...]] = []
    for k in range(1, max(1, max_size) + 1):
        combos.extend(combinations(targets, k))
    return combos


def bliss_synergy(effect_a: float, effect_b: float, effect_ab: float) -> float:
    """Bliss excess synergy: Eab - (Ea + Eb - Ea*Eb)."""
    expected = effect_a + effect_b - effect_a * effect_b
    return float(effect_ab - expected)


def rank_combinations(
    combo_effects: dict[tuple[str, ...], float],
    single_effects: dict[str, float],
) -> list[dict]:
    """Rank combinations by effect then synergy."""
    rows = []
    for combo, eff in combo_effects.items():
        combo = tuple(combo)
        if len(combo) == 2:
            ea = single_effects.get(combo[0], 0.0)
            eb = single_effects.get(combo[1], 0.0)
            syn = bliss_synergy(ea, eb, eff)
        else:
            syn = 0.0
        rows.append(
            {
                "combo": "+".join(combo),
                "size": len(combo),
                "effect": float(eff),
                "synergy_bliss": float(syn),
            }
        )
    rows.sort(key=lambda x: (x["effect"], x["synergy_bliss"]), reverse=True)
    return rows
