from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest
import torch

from src.causal.hypersca_c import (
    HyperSCACConfig,
    HyperSCACError,
    build_intervention_mask,
    masked_sem_loss,
    zero_diagonal,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CONFIG_PAYLOAD: dict[str, object] = {
    "schema_version": "1.0",
    "learning_rate": 0.01,
    "maximum_epochs": 200,
    "early_stopping_patience": 10,
    "shared_l1": 0.001,
    "context_l1": 0.002,
    "acyclicity_weight": 0.01,
    "enable_context_adjustments": True,
    "prior_discount": 0.0,
    "selection_threshold": 0.0001,
    "bootstrap_repeats": 20,
    "bootstrap_success_fraction": 0.8,
    "minimum_source_variance": 1e-08,
    "control_label": "non-targeting",
    "excluded_label": "excluded",
}


def default_config_payload() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/hypersca_c_v1.json").read_text(encoding="utf-8")
    )


def test_default_config_is_frozen_and_has_no_prior_discount() -> None:
    payload = default_config_payload()
    assert payload == EXPECTED_CONFIG_PAYLOAD
    config = HyperSCACConfig.from_mapping(payload)
    assert config.schema_version == "1.0"
    assert config.prior_discount == 0.0
    assert config.enable_context_adjustments is True
    assert config.bootstrap_success_fraction == 0.8


def test_intervention_mask_excludes_only_the_direct_target() -> None:
    mask = build_intervention_mask(
        ["non-targeting", "B", "excluded"],
        ["A", "B", "C"],
    )
    assert mask.tolist() == [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]


def test_masked_loss_ignores_perturbed_gene_self_error() -> None:
    observed = torch.tensor([[1.0, 7.0]])
    prediction = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert float(masked_sem_loss(prediction, observed, mask)) == pytest.approx(0.0)


def test_zero_diagonal_removes_self_edges() -> None:
    matrix = torch.ones((3, 3))
    result = zero_diagonal(matrix)
    assert torch.diagonal(result).tolist() == [0.0, 0.0, 0.0]
    assert float(result.sum()) == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"schema_version": "2.0"}, "schema_version"),
        ({"schema_version": 1.0}, "schema_version"),
        ({"learning_rate": True}, "learning_rate"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"maximum_epochs": True}, "maximum_epochs"),
        ({"maximum_epochs": 0}, "maximum_epochs"),
        ({"early_stopping_patience": 0}, "early_stopping_patience"),
        ({"shared_l1": -0.1}, "shared_l1"),
        ({"context_l1": float("inf")}, "context_l1"),
        ({"acyclicity_weight": -0.1}, "acyclicity_weight"),
        ({"enable_context_adjustments": 1}, "enable_context_adjustments"),
        ({"prior_discount": 1.0}, "prior_discount"),
        ({"selection_threshold": -0.1}, "selection_threshold"),
        ({"bootstrap_repeats": 1.5}, "bootstrap_repeats"),
        ({"bootstrap_success_fraction": 0.0}, "bootstrap_success_fraction"),
        ({"minimum_source_variance": 0.0}, "minimum_source_variance"),
        ({"control_label": ""}, "control_label"),
        ({"excluded_label": " excluded"}, "excluded_label"),
        ({"control_label": "non targeting"}, "control_label"),
        ({"excluded_label": "excl\tuded"}, "excluded_label"),
        ({"control_label": "non\ntargeting"}, "control_label"),
        ({"excluded_label": "excl\u2003uded"}, "excluded_label"),
        (
            {"control_label": "excluded", "excluded_label": "excluded"},
            "different",
        ),
    ],
)
def test_config_rejects_invalid_frozen_values(
    change: dict[str, object], match: str
) -> None:
    payload = default_config_payload()
    payload.update(change)
    with pytest.raises(HyperSCACError, match=match):
        HyperSCACConfig.from_mapping(payload)


@pytest.mark.parametrize("edit", ["missing", "extra"])
def test_config_requires_exactly_the_frozen_fields(edit: str) -> None:
    payload = default_config_payload()
    if edit == "missing":
        del payload["learning_rate"]
    else:
        payload["unexpected_setting"] = 1
    with pytest.raises(HyperSCACError, match="fields"):
        HyperSCACConfig.from_mapping(payload)


def test_config_wraps_non_mapping_input_in_domain_error() -> None:
    with pytest.raises(HyperSCACError, match="mapping"):
        HyperSCACConfig.from_mapping(  # type: ignore[arg-type]
            [("learning_rate", 0.01)]
        )


def test_config_wraps_mapping_read_failure_in_domain_error() -> None:
    class UnreadableValues(Mapping[str, object]):
        def __iter__(self) -> Iterator[str]:
            return iter(EXPECTED_CONFIG_PAYLOAD)

        def __len__(self) -> int:
            return len(EXPECTED_CONFIG_PAYLOAD)

        def __getitem__(self, key: str) -> object:
            raise RuntimeError(f"cannot read {key}")

    with pytest.raises(HyperSCACError, match="validated"):
        HyperSCACConfig.from_mapping(UnreadableValues())


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"schema_version": "2.0"}, "schema_version"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"shared_l1": -0.1}, "shared_l1"),
        ({"bootstrap_repeats": 0}, "bootstrap_repeats"),
        ({"enable_context_adjustments": 1}, "enable_context_adjustments"),
        ({"control_label": ""}, "control_label"),
        ({"excluded_label": "excl\tuded"}, "excluded_label"),
    ],
)
def test_direct_config_construction_cannot_bypass_validation(
    change: dict[str, object], match: str
) -> None:
    payload = dict(EXPECTED_CONFIG_PAYLOAD)
    payload.update(change)
    with pytest.raises(HyperSCACError, match=match):
        HyperSCACConfig(**payload)  # type: ignore[arg-type]


def test_direct_config_construction_normalizes_numpy_numbers() -> None:
    payload = dict(EXPECTED_CONFIG_PAYLOAD)
    payload.update(
        {
            "learning_rate": np.float32(0.01),
            "maximum_epochs": np.int64(200),
            "early_stopping_patience": np.int32(10),
            "bootstrap_repeats": np.int64(20),
        }
    )
    config = HyperSCACConfig(**payload)  # type: ignore[arg-type]
    assert type(config.learning_rate) is float
    assert type(config.maximum_epochs) is int
    assert type(config.early_stopping_patience) is int
    assert type(config.bootstrap_repeats) is int


def test_intervention_mask_keeps_unknown_but_valid_labels_available() -> None:
    mask = build_intervention_mask(["unknown-target"], ["A", "B"])
    assert mask.dtype == np.float32
    assert mask.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize(
    ("interventions", "gene_names", "match"),
    [
        (["A"], [], "gene_names"),
        (["A"], ["A", "A"], "unique"),
        (["A"], [" A"], "whitespace"),
        (["A"], np.asarray([["A"]]), "one-dimensional"),
        ([1], ["A"], "interventions"),
        ([" A"], ["A"], "whitespace"),
        (["unknown target"], ["A"], "whitespace"),
        (["unknown\ttarget"], ["A"], "whitespace"),
        (["unknown\ntarget"], ["A"], "whitespace"),
        (["unknown\u2003target"], ["A"], "whitespace"),
        (["A"], ["gene name"], "whitespace"),
        (["A"], ["gene\tname"], "whitespace"),
        (["A"], ["gene\nname"], "whitespace"),
        (["A"], ["gene\u2003name"], "whitespace"),
        ([], ["A"], "interventions"),
        (["excluded"], ["A"], "no usable"),
    ],
)
def test_intervention_mask_rejects_ambiguous_inputs(
    interventions: object,
    gene_names: object,
    match: str,
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        build_intervention_mask(interventions, gene_names)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "excluded_label",
    ["excluded label", "excluded\tlabel", "excluded\nlabel", "excluded\u2003label"],
)
def test_intervention_mask_rejects_whitespace_in_excluded_label(
    excluded_label: str,
) -> None:
    with pytest.raises(HyperSCACError, match="excluded_label.*whitespace"):
        build_intervention_mask(["A"], ["A"], excluded_label=excluded_label)


def test_zero_diagonal_preserves_tensor_properties_and_gradient() -> None:
    matrix = torch.ones((2, 2), dtype=torch.float64, requires_grad=True)
    before = matrix.detach().clone()
    result = zero_diagonal(matrix)
    result.sum().backward()
    assert result.dtype == matrix.dtype
    assert result.device == matrix.device
    assert torch.equal(matrix.detach(), before)
    assert matrix.grad is not None
    assert matrix.grad.tolist() == [[0.0, 1.0], [1.0, 0.0]]


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        ([[1.0]], "tensor"),
        (torch.ones(2), "two-dimensional"),
        (torch.ones((2, 3)), "square"),
        (torch.ones((2, 2), dtype=torch.int64), "floating-point"),
    ],
)
def test_zero_diagonal_rejects_invalid_matrices(matrix: object, match: str) -> None:
    with pytest.raises(HyperSCACError, match=match):
        zero_diagonal(matrix)  # type: ignore[arg-type]


def test_masked_loss_normalizes_only_over_available_values_and_keeps_gradient() -> None:
    prediction = torch.tensor([[2.0, 100.0], [0.0, 2.0]], requires_grad=True)
    observed = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    loss = masked_sem_loss(prediction, observed, mask)
    loss.backward()
    assert loss.detach().item() == pytest.approx(1.0)
    assert prediction.grad is not None
    assert prediction.grad[0, 1].item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_masked_loss_promotes_low_precision_before_reduction(
    dtype: torch.dtype,
) -> None:
    prediction = torch.ones(70_000, dtype=dtype, requires_grad=True)
    observed = torch.zeros(70_000, dtype=dtype)
    mask = torch.ones(70_000, dtype=dtype)
    loss = masked_sem_loss(prediction, observed, mask)
    loss.backward()
    assert loss.dtype == torch.float32
    assert loss.detach().item() == pytest.approx(0.5)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_masked_loss_never_computes_error_for_hidden_extreme_values() -> None:
    largest = torch.finfo(torch.float32).max
    prediction = torch.tensor([[1.0, largest]], requires_grad=True)
    observed = torch.tensor([[0.0, -largest]])
    mask = torch.tensor([[1.0, 0.0]])
    loss = masked_sem_loss(prediction, observed, mask)
    loss.backward()
    assert loss.detach().item() == pytest.approx(0.5)
    assert prediction.grad is not None
    assert prediction.grad.tolist() == [[1.0, 0.0]]


def test_masked_loss_rejects_nonfinite_result_from_visible_extreme_values() -> None:
    largest = torch.finfo(torch.float32).max
    with pytest.raises(HyperSCACError, match="loss.*finite"):
        masked_sem_loss(
            torch.tensor([[largest]]),
            torch.tensor([[-largest]]),
            torch.tensor([[1.0]]),
        )


def test_masked_loss_rejects_trainable_experimental_mask() -> None:
    with pytest.raises(HyperSCACError, match="mask.*gradient"):
        masked_sem_loss(
            torch.ones((1, 1)),
            torch.zeros((1, 1)),
            torch.ones((1, 1), requires_grad=True),
        )


def test_masked_loss_supports_non_contiguous_tensors() -> None:
    prediction_base = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    prediction = prediction_base.transpose(0, 1)
    observed = torch.zeros((2, 3)).transpose(0, 1)
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).transpose(0, 1)
    assert not prediction.is_contiguous()
    assert not mask.is_contiguous()
    loss = masked_sem_loss(prediction, observed, mask)
    expected = torch.nn.functional.smooth_l1_loss(
        prediction[mask == 1.0], observed[mask == 1.0], reduction="mean"
    )
    assert loss.detach().item() == pytest.approx(expected.detach().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_masked_loss_matches_cpu_on_cuda() -> None:
    prediction = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    observed = torch.zeros_like(prediction)
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    cpu_loss = masked_sem_loss(prediction, observed, mask)
    cuda_loss = masked_sem_loss(
        prediction.cuda(), observed.cuda(), mask.cuda()
    ).cpu()
    assert cuda_loss.item() == pytest.approx(cpu_loss.item())


def test_masked_loss_rejects_sparse_tensors_before_numeric_operations() -> None:
    with pytest.raises(HyperSCACError, match="dense"):
        masked_sem_loss(
            torch.ones((1, 1)).to_sparse(),
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
        )


def test_masked_loss_rejects_complex_and_meta_tensors() -> None:
    with pytest.raises(HyperSCACError, match="supported.*dtype"):
        masked_sem_loss(
            torch.ones((1, 1), dtype=torch.complex64),
            torch.zeros((1, 1), dtype=torch.complex64),
            torch.ones((1, 1), dtype=torch.complex64),
        )
    with pytest.raises(HyperSCACError, match="materialized"):
        masked_sem_loss(
            torch.ones((1, 1), device="meta"),
            torch.zeros((1, 1), device="meta"),
            torch.ones((1, 1), device="meta"),
        )


FLOAT8_DTYPES = tuple(
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, name, None)) is not None
)


@pytest.mark.parametrize("dtype", FLOAT8_DTYPES)
def test_masked_loss_rejects_float8_without_leaking_torch_errors(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(HyperSCACError, match="supported.*dtype"):
        masked_sem_loss(
            torch.zeros((1, 1), dtype=dtype),
            torch.zeros((1, 1), dtype=dtype),
            torch.zeros((1, 1), dtype=dtype),
        )


@pytest.mark.parametrize(
    ("prediction", "observed", "mask", "match"),
    [
        ([[1.0]], torch.ones((1, 1)), torch.ones((1, 1)), "tensor"),
        (torch.ones((1, 2)), torch.ones((1, 1)), torch.ones((1, 1)), "shapes"),
        (torch.empty((0, 1)), torch.empty((0, 1)), torch.empty((0, 1)), "empty"),
        (
            torch.ones((1, 1), dtype=torch.int64),
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            "floating-point",
        ),
        (
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            torch.ones((1, 1), dtype=torch.int64),
            "floating-point",
        ),
        (
            torch.tensor([[float("nan")]]),
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            "finite",
        ),
        (torch.ones((1, 1)), torch.ones((1, 1)), torch.tensor([[0.5]]), "0 or 1"),
        (torch.ones((1, 1)), torch.ones((1, 1)), torch.zeros((1, 1)), "usable"),
    ],
)
def test_masked_loss_rejects_invalid_inputs(
    prediction: object,
    observed: torch.Tensor,
    mask: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        masked_sem_loss(prediction, observed, mask)  # type: ignore[arg-type]


def test_masked_loss_rejects_tensors_on_different_devices() -> None:
    with pytest.raises(HyperSCACError, match="device"):
        masked_sem_loss(
            torch.ones((1, 1)),
            torch.ones((1, 1), device="meta"),
            torch.ones((1, 1)),
        )
