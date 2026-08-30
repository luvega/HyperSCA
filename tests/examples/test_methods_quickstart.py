from __future__ import annotations

import numpy as np
import pytest

from src.examples.methods_quickstart import make_hypersca_methods_example


def test_default_methods_example_has_frozen_scientific_contract() -> None:
    example = make_hypersca_methods_example()

    assert example.expression.shape == (512, 128)
    assert example.positions.shape == (512, 2)
    assert example.causal_adjacency.shape == (128, 128)
    assert len(example.gene_names) == 128
    assert len(example.cell_types) == 512
    assert len(example.interventions) == 512
    assert len(example.spatial_blocks) == 512
    assert np.isfinite(example.expression).all()
    assert np.all(example.expression >= 0)
    assert np.isfinite(example.positions).all()
    assert np.allclose(np.diag(example.causal_adjacency), 0.0)
    assert np.allclose(np.tril(example.causal_adjacency), 0.0)
    assert int(np.count_nonzero(example.causal_adjacency)) > 0
    assert set(example.interventions) <= {"control", *example.gene_names}
    assert set(example.interventions) - {"control"}


def test_methods_example_is_seeded_without_touching_global_numpy_state() -> None:
    np.random.seed(991)
    expected_next = np.random.random()
    np.random.seed(991)

    first = make_hypersca_methods_example(seed=17)
    second = make_hypersca_methods_example(seed=17)
    third = make_hypersca_methods_example(seed=18)
    actual_next = np.random.random()

    assert actual_next == expected_next
    assert np.array_equal(first.expression, second.expression)
    assert np.array_equal(first.positions, second.positions)
    assert first.cell_types == second.cell_types
    assert first.interventions == second.interventions
    assert not np.array_equal(first.expression, third.expression)


def test_methods_example_arrays_are_hard_read_only() -> None:
    example = make_hypersca_methods_example(n_cells=64, n_genes=32)

    for array in (example.expression, example.positions, example.causal_adjacency):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_cells": 31}, "n_cells"),
        ({"n_genes": 15}, "n_genes"),
        ({"seed": True}, "seed"),
    ],
)
def test_methods_example_rejects_invalid_contract_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_hypersca_methods_example(**kwargs)
