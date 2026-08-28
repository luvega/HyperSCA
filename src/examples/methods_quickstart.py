"""Deterministic public toy data for the HyperSCA methods quickstart.

The example is deliberately small and self-contained.  It contains spatial
blocks, a simple cell-type hierarchy, single-gene interventions, and a known
directed acyclic graph.  It is suitable for tutorials and continuous
integration, not for estimating biological performance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIN_CELLS = 32
MAX_CELLS = 20_000
MIN_GENES = 16
MAX_GENES = 1_000
CELL_TYPE_BY_BLOCK = ("Tumor", "T_cell", "Myeloid", "Fibroblast")


def _hard_read_only(array: np.ndarray, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class MethodsQuickstartData:
    """Immutable arrays and labels used by the methods quickstart."""

    expression: np.ndarray
    positions: np.ndarray
    cell_types: tuple[str, ...]
    interventions: tuple[str, ...]
    gene_names: tuple[str, ...]
    causal_adjacency: np.ndarray
    spatial_blocks: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0 or self.seed > 2**32 - 1:
            raise ValueError("seed must be a non-negative built-in integer no larger than 2**32 - 1")
        expression = np.asarray(self.expression)
        positions = np.asarray(self.positions)
        adjacency = np.asarray(self.causal_adjacency)
        if expression.ndim != 2 or expression.shape != (len(self.cell_types), len(self.gene_names)):
            raise ValueError("expression shape must match cell_types and gene_names")
        if positions.shape != (len(self.cell_types), 2):
            raise ValueError("positions must have shape (n_cells, 2)")
        if adjacency.shape != (len(self.gene_names), len(self.gene_names)):
            raise ValueError("causal_adjacency shape must match gene_names")
        if len(self.interventions) != len(self.cell_types) or len(self.spatial_blocks) != len(self.cell_types):
            raise ValueError("cell-level labels must match the expression rows")
        if not np.isfinite(expression).all() or np.any(expression < 0):
            raise ValueError("expression must contain finite non-negative values")
        if not np.isfinite(positions).all():
            raise ValueError("positions must contain finite values")
        if not np.isfinite(adjacency).all() or np.any(adjacency < 0):
            raise ValueError("causal_adjacency must contain finite non-negative values")
        if np.any(np.diag(adjacency)) or np.any(np.tril(adjacency)):
            raise ValueError("causal_adjacency must be a strictly upper-triangular DAG")
        if not set(self.interventions) <= {"control", *self.gene_names}:
            raise ValueError("interventions must be control or a measured gene")

        object.__setattr__(self, "expression", _hard_read_only(expression, dtype=np.dtype("<f8")))
        object.__setattr__(self, "positions", _hard_read_only(positions, dtype=np.dtype("<f8")))
        object.__setattr__(self, "causal_adjacency", _hard_read_only(adjacency, dtype=np.dtype("<f8")))


def _validate_size(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a built-in integer in [{minimum}, {maximum}]")
    return value


def _known_dag(n_genes: int) -> np.ndarray:
    adjacency = np.zeros((n_genes, n_genes), dtype=np.float64)
    regulator_count = min(8, max(2, n_genes // 8))
    first_target = regulator_count
    for source in range(regulator_count):
        for offset in range(3):
            target = first_target + source * 3 + offset
            if target < n_genes:
                adjacency[source, target] = 1.0 - 0.2 * offset
    for source in range(first_target, min(n_genes - 1, first_target + regulator_count * 2)):
        target = source + regulator_count * 2
        if target < n_genes:
            adjacency[source, target] = 0.4
    return adjacency


def make_hypersca_methods_example(
    *,
    seed: int = 11,
    n_cells: int = 512,
    n_genes: int = 128,
) -> MethodsQuickstartData:
    """Build deterministic toy data with spatial, hierarchical, and causal signal.

    The local random generator is isolated from NumPy's global random state.
    Perturbations are simulated knock-downs of upstream genes in the known DAG.
    """

    if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be a non-negative built-in integer no larger than 2**32 - 1")
    n_cells = _validate_size(n_cells, name="n_cells", minimum=MIN_CELLS, maximum=MAX_CELLS)
    n_genes = _validate_size(n_genes, name="n_genes", minimum=MIN_GENES, maximum=MAX_GENES)
    rng = np.random.default_rng(seed)

    width = int(np.ceil(np.sqrt(n_cells)))
    row = np.arange(n_cells) // width
    column = np.arange(n_cells) % width
    row_scale = max(int(row.max()), 1)
    column_scale = max(width - 1, 1)
    positions = np.column_stack((column / column_scale, row / row_scale)).astype(np.float64)
    positions += rng.normal(0.0, 0.004, size=positions.shape)
    positions = np.clip(positions, 0.0, 1.0)
    blocks = (positions[:, 0] >= 0.5).astype(int) + 2 * (positions[:, 1] >= 0.5).astype(int)
    cell_types = tuple(CELL_TYPE_BY_BLOCK[int(block)] for block in blocks)

    gene_names = tuple(f"G{index:03d}" for index in range(n_genes))
    adjacency = _known_dag(n_genes)
    expression = rng.lognormal(mean=0.3, sigma=0.25, size=(n_cells, n_genes))

    block_width = max(2, n_genes // 16)
    for block in range(4):
        start = block * block_width
        stop = min(start + block_width, n_genes)
        expression[blocks == block, start:stop] += 2.0

    for source, target in zip(*np.nonzero(adjacency), strict=False):
        expression[:, target] += adjacency[source, target] * np.log1p(expression[:, source])

    regulator_count = min(8, max(2, n_genes // 8))
    interventions = np.full(n_cells, "control", dtype=object)
    perturbed_rows = np.arange(4, n_cells, 5)
    assigned_sources = np.arange(len(perturbed_rows)) % regulator_count
    rng.shuffle(assigned_sources)
    for row_index, source in zip(perturbed_rows, assigned_sources, strict=True):
        interventions[row_index] = gene_names[int(source)]
        expression[row_index, source] *= 0.15
        downstream = np.flatnonzero(adjacency[source] > 0)
        expression[row_index, downstream] *= 0.65

    expression += rng.gamma(shape=1.0, scale=0.02, size=expression.shape)

    return MethodsQuickstartData(
        expression=expression,
        positions=positions,
        cell_types=cell_types,
        interventions=tuple(str(value) for value in interventions),
        gene_names=gene_names,
        causal_adjacency=adjacency,
        spatial_blocks=tuple(int(value) for value in blocks),
        seed=seed,
    )


__all__ = ["MethodsQuickstartData", "make_hypersca_methods_example"]
