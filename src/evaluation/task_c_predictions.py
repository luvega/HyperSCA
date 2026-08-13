"""把各方法返回的关系映射到相同、完整的有向关系范围。"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
import unicodedata

import numpy as np
import pandas as pd


MAXIMUM_TASK_C_GENES = 1_000
"""本层允许展开的基因上限；正式演练可在外层使用更小上限。"""


class TaskCPredictionError(ValueError):
    """一个方法的关系结果不能安全进入统一评价。"""


def _is_canonical_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and unicodedata.is_normalized("NFC", value)
    )


def _validated_genes(gene_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(gene_names, (str, bytes)) or not isinstance(gene_names, Sequence):
        raise TaskCPredictionError("gene names must be provided as an ordered sequence")
    if len(gene_names) == 0:
        raise TaskCPredictionError("gene names must not be empty")
    if len(gene_names) > MAXIMUM_TASK_C_GENES:
        raise TaskCPredictionError(
            f"gene names must contain at most {MAXIMUM_TASK_C_GENES} entries"
        )

    genes = tuple(gene_names)
    if any(not _is_canonical_name(gene) for gene in genes):
        raise TaskCPredictionError(
            "gene names must be non-empty NFC strings without surrounding whitespace"
        )
    if len(set(genes)) != len(genes):
        raise TaskCPredictionError("gene names must be unique")
    return genes


def _validated_scores(values: list[object]) -> np.ndarray:
    scores: list[float] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, np.integer, np.floating)
        ):
            raise TaskCPredictionError(
                "scores must be real, finite, non-negative numbers"
            )
        try:
            score = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise TaskCPredictionError(
                "scores must be real, finite, non-negative numbers"
            ) from exc
        if not np.isfinite(score) or score < 0.0:
            raise TaskCPredictionError(
                "scores must be real, finite, non-negative numbers"
            )
        scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def normalize_task_c_predictions(
    raw: pd.DataFrame,
    gene_names: Sequence[str],
) -> pd.DataFrame:
    """补齐未返回的关系，同时保留方法是否实际返回过每条关系。"""
    genes = _validated_genes(gene_names)
    if not isinstance(raw, pd.DataFrame):
        raise TaskCPredictionError("prediction table must be a pandas DataFrame")
    required_columns = {"source", "target", "score"}
    if len(raw.columns) != 3 or set(raw.columns) != required_columns:
        raise TaskCPredictionError(
            "prediction table must contain exactly source, target, and score"
        )

    sources = raw["source"].tolist()
    targets = raw["target"].tolist()
    if any(not _is_canonical_name(value) for value in sources + targets):
        raise TaskCPredictionError(
            "relation endpoints must be NFC strings without surrounding whitespace"
        )
    gene_set = set(genes)
    if any(source not in gene_set for source in sources) or any(
        target not in gene_set for target in targets
    ):
        raise TaskCPredictionError(
            "prediction table contains a gene outside the fixed gene set"
        )
    scores = _validated_scores(raw["score"].tolist())

    selected = pd.DataFrame(
        {
            "source": pd.array(sources, dtype="string"),
            "target": pd.array(targets, dtype="string"),
            "score": scores,
        }
    )
    selected = selected[selected["source"] != selected["target"]].reset_index(
        drop=True
    )
    selected = selected.groupby(
        ["source", "target"], as_index=False, sort=False
    )["score"].max()
    selected["_returned"] = True

    gene_array = np.asarray(genes, dtype=object)
    source_values = np.repeat(gene_array, len(genes))
    target_values = np.tile(gene_array, len(genes))
    non_self = source_values != target_values
    universe = pd.DataFrame(
        {
            "source": pd.array(source_values[non_self], dtype="string"),
            "target": pd.array(target_values[non_self], dtype="string"),
            "_fixed_order": np.arange(int(non_self.sum()), dtype=np.int64),
        }
    )

    completed = universe.merge(
        selected,
        how="left",
        on=["source", "target"],
        sort=False,
        validate="one_to_one",
    )
    completed["score"] = completed["score"].fillna(0.0).astype(np.float64)
    completed["returned_by_method"] = completed["_returned"].eq(True)
    completed = completed.sort_values(
        ["score", "_fixed_order"],
        ascending=[False, True],
        kind="mergesort",
    ).drop(columns=["_fixed_order", "_returned"])
    return completed[
        ["source", "target", "score", "returned_by_method"]
    ].reset_index(drop=True)
