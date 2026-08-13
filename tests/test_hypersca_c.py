from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest
import torch

from src.causal.hypersca_c import (
    HyperSCACConfig,
    HyperSCACContext,
    HyperSCACError,
    HyperSCACFit,
    acyclicity_penalty,
    build_intervention_mask,
    fit_hypersca_c_once,
    masked_sem_loss,
    standardize_context,
    zero_diagonal,
)
from src.causal import hypersca_c as hypersca_c_module


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


def small_config(**changes: object) -> HyperSCACConfig:
    payload = default_config_payload()
    payload.update(
        {
            "maximum_epochs": 8,
            "early_stopping_patience": 8,
            "acyclicity_weight": 0.001,
            **changes,
        }
    )
    return HyperSCACConfig.from_mapping(payload)


def small_context(
    context_id: str = "k562",
    *,
    gene_names: tuple[str, ...] = ("A", "B"),
) -> HyperSCACContext:
    expression = np.asarray(
        [
            [-1.0, -1.1],
            [1.0, 1.1],
            [-0.5, -0.6],
            [0.5, 0.6],
            [-1.5, -1.8],
            [1.5, 1.8],
        ],
        dtype=np.float64,
    )
    return HyperSCACContext(
        context_id=context_id,
        expression=expression,
        interventions=np.asarray(
            ["non-targeting"] * 4 + ["A", "A"], dtype=object
        ),
        gene_names=gene_names,
    )


def test_context_owns_read_only_normalized_copies() -> None:
    expression = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    interventions = np.asarray(["non-targeting", "A"], dtype=object)
    context = HyperSCACContext(
        context_id="k562",
        expression=expression,
        interventions=interventions,
        gene_names=("A", "B"),
    )
    expression[0, 0] = 99
    interventions[0] = "changed"
    assert context.expression.dtype == np.float32
    assert context.expression[0, 0] == 0.0
    assert context.interventions.tolist() == ["non-targeting", "A"]
    assert context.expression.flags.writeable is False
    assert context.interventions.flags.writeable is False
    with pytest.raises(ValueError):
        context.expression[0, 0] = 2.0


def test_context_arrays_cannot_have_write_permission_restored() -> None:
    context = small_context()
    assert context.interventions.dtype.kind == "U"
    assert context.interventions.shape == (6,)
    for public_array in (context.expression, context.interventions):
        assert public_array.flags.writeable is False
        with pytest.raises(ValueError):
            public_array.setflags(write=True)


@pytest.mark.parametrize(
    ("context_id", "expression", "interventions", "gene_names", "match"),
    [
        ("bad id", np.ones((2, 2)), ["non-targeting"] * 2, ("A", "B"), "context_id"),
        ("k562", np.ones(2), ["non-targeting"] * 2, ("A", "B"), "two-dimensional"),
        ("k562", np.empty((0, 2)), [], ("A", "B"), "at least one"),
        ("k562", np.ones((2, 2), dtype=bool), ["non-targeting"] * 2, ("A", "B"), "numeric"),
        ("k562", np.asarray([[1.0, np.nan], [2.0, 3.0]]), ["non-targeting"] * 2, ("A", "B"), "finite"),
        ("k562", np.ones((2, 3)), ["non-targeting"] * 2, ("A", "B"), "shape"),
        ("k562", np.ones((2, 2)), [1, 2], ("A", "B"), "interventions"),
        ("k562", np.ones((2, 2)), ["non-targeting", "bad label"], ("A", "B"), "whitespace"),
        ("k562", np.ones((2, 2)), ["non-targeting"] * 2, ("A", "A"), "unique"),
        ("k562", np.ones((2, 1)), ["non-targeting"] * 2, tuple(), "gene_names"),
    ],
)
def test_context_rejects_ambiguous_or_inconsistent_inputs(
    context_id: object,
    expression: object,
    interventions: object,
    gene_names: object,
    match: str,
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        HyperSCACContext(
            context_id=context_id,  # type: ignore[arg-type]
            expression=expression,  # type: ignore[arg-type]
            interventions=interventions,  # type: ignore[arg-type]
            gene_names=gene_names,  # type: ignore[arg-type]
        )


def test_standardization_uses_control_cells_only_and_population_scale() -> None:
    expression = np.asarray(
        [[0.0, 5.0], [2.0, 5.0], [100.0, 100.0]], dtype=np.float64
    )
    before = expression.copy()
    scaled, center, scale = standardize_context(
        expression,
        ["non-targeting", "non-targeting", "A"],
        control_label="non-targeting",
    )
    assert center.tolist() == pytest.approx([1.0, 5.0])
    assert scale.tolist() == pytest.approx([1.0, 1.0])
    assert scaled[:2].mean(axis=0).tolist() == pytest.approx([0.0, 0.0])
    assert scaled.dtype == center.dtype == scale.dtype == np.float32
    assert np.isfinite(scaled).all()
    assert np.array_equal(expression, before)


@pytest.mark.parametrize(
    ("expression", "labels", "control_label", "match"),
    [
        (np.ones(3), ["non-targeting"] * 3, "non-targeting", "two-dimensional"),
        (np.ones((2, 2), dtype=bool), ["non-targeting"] * 2, "non-targeting", "numeric"),
        (np.asarray([[1.0, np.inf], [2.0, 3.0]]), ["non-targeting"] * 2, "non-targeting", "finite"),
        (np.ones((2, 2)), ["non-targeting"], "non-targeting", "rows"),
        (np.ones((2, 2)), [1, 1], "1", "interventions"),
        (np.ones((2, 2)), ["non-targeting", "A"], "non-targeting", "two control"),
        (np.ones((2, 2)), ["non-targeting"] * 2, "control label", "control_label"),
    ],
)
def test_standardization_rejects_invalid_inputs(
    expression: object,
    labels: object,
    control_label: object,
    match: str,
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        standardize_context(
            expression,  # type: ignore[arg-type]
            labels,  # type: ignore[arg-type]
            control_label=control_label,  # type: ignore[arg-type]
        )


def test_acyclicity_penalty_distinguishes_a_dag_from_a_two_cycle() -> None:
    dag = torch.tensor([[0.0, 1.2], [0.0, 0.0]], requires_grad=True)
    cycle = torch.tensor([[0.0, 1.2], [0.7, 0.0]])
    dag_penalty = acyclicity_penalty(dag)
    cycle_penalty = acyclicity_penalty(cycle)
    dag_penalty.backward()
    assert dag_penalty.item() == pytest.approx(0.0, abs=1e-6)
    assert cycle_penalty.item() > 0.0
    assert dag.grad is not None
    assert torch.isfinite(dag.grad).all()


@pytest.mark.parametrize(
    ("adjacency", "match"),
    [
        ([[0.0]], "tensor"),
        (torch.empty((0, 0)), "at least one"),
        (torch.ones(2), "two-dimensional"),
        (torch.ones((2, 3)), "square"),
        (torch.ones((2, 2), dtype=torch.int64), "floating-point"),
        (torch.ones((2, 2), dtype=torch.complex64), "supported"),
        (torch.tensor([[0.0, float("nan")], [0.0, 0.0]]), "finite"),
        (torch.ones((2, 2)).to_sparse(), "dense"),
        (torch.ones((2, 2), device="meta"), "materialized"),
    ],
)
def test_acyclicity_penalty_wraps_invalid_tensor_inputs(
    adjacency: object, match: str
) -> None:
    with pytest.raises(HyperSCACError, match=match):
        acyclicity_penalty(adjacency)  # type: ignore[arg-type]


def test_joint_fit_returns_immutable_shared_and_context_specific_matrices() -> None:
    contexts = [small_context("k562"), small_context("rpe1")]
    config = small_config()
    result = fit_hypersca_c_once(
        contexts, config, seed=11, device="cpu"
    )
    assert result.shared.shape == (2, 2)
    assert list(result.context_adjustments) == ["k562", "rpe1"]
    assert list(result.context_adjacencies) == ["k562", "rpe1"]
    assert np.diag(result.shared).tolist() == pytest.approx([0.0, 0.0])
    assert np.isfinite(result.loss_history).all()
    assert result.epochs_run == len(result.loss_history)
    assert 1 <= result.epochs_run <= config.maximum_epochs
    assert result.config is config
    for name in ("k562", "rpe1"):
        assert np.array_equal(
            result.context_adjacencies[name],
            result.shared + result.context_adjustments[name],
        )
        assert np.diag(result.context_adjacencies[name]).tolist() == pytest.approx(
            [0.0, 0.0]
        )
        assert result.context_adjustments[name].flags.writeable is False
        assert result.context_adjacencies[name].flags.writeable is False
    assert result.shared.flags.writeable is False
    assert result.loss_history.flags.writeable is False
    with pytest.raises(TypeError):
        result.context_adjustments["new"] = np.zeros((2, 2))  # type: ignore[index]


def test_disabled_context_adjustments_are_exactly_zero_and_not_optimized() -> None:
    result = fit_hypersca_c_once(
        [small_context("k562"), small_context("rpe1")],
        small_config(enable_context_adjustments=False),
        seed=3,
        device="cpu",
    )
    for name in result.context_adjustments:
        assert np.array_equal(result.context_adjustments[name], np.zeros((2, 2)))
        assert np.array_equal(result.context_adjacencies[name], result.shared)


def test_fit_preserves_global_random_number_generator_states() -> None:
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    fit_hypersca_c_once(
        [small_context()], small_config(maximum_epochs=2), seed=11, device="cpu"
    )
    after_np_state = np.random.get_state()
    assert np_state[0] == after_np_state[0]
    assert np.array_equal(np_state[1], after_np_state[1])
    assert np_state[2:] == after_np_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_same_seed_repeats_the_same_fit() -> None:
    contexts = [small_context("k562"), small_context("rpe1")]
    first = fit_hypersca_c_once(contexts, small_config(), seed=47, device="cpu")
    second = fit_hypersca_c_once(contexts, small_config(), seed=47, device="cpu")
    assert np.array_equal(first.shared, second.shared)
    assert np.array_equal(first.loss_history, second.loss_history)
    for name in first.context_adjustments:
        assert np.array_equal(
            first.context_adjustments[name], second.context_adjustments[name]
        )


def test_prior_discount_zero_makes_binary_prior_irrelevant() -> None:
    context = [small_context()]
    without_prior = fit_hypersca_c_once(
        context, small_config(prior_discount=0.0), seed=11, device="cpu"
    )
    with_prior = fit_hypersca_c_once(
        context,
        small_config(prior_discount=0.0),
        seed=11,
        device="cpu",
        prior_mask=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    )
    assert np.array_equal(without_prior.shared, with_prior.shared)
    assert np.array_equal(without_prior.loss_history, with_prior.loss_history)


def test_diagonal_prior_values_do_not_change_the_fit() -> None:
    config = small_config(prior_discount=0.5)
    context = [small_context()]
    zero_prior = fit_hypersca_c_once(
        context, config, seed=11, device="cpu", prior_mask=np.zeros((2, 2))
    )
    diagonal_prior = fit_hypersca_c_once(
        context, config, seed=11, device="cpu", prior_mask=np.eye(2)
    )
    assert np.array_equal(zero_prior.shared, diagonal_prior.shared)
    assert np.array_equal(zero_prior.loss_history, diagonal_prior.loss_history)


@pytest.mark.parametrize(
    ("prior", "match"),
    [
        (np.ones((2, 3)), "shape"),
        (np.asarray([[0, 2], [1, 0]]), "0 or 1"),
        (np.asarray([[0.0, np.nan], [1.0, 0.0]]), "finite"),
        (np.asarray([["0", "1"], ["1", "0"]]), "numeric"),
        (np.asarray([[0.0 + 0.0j, 1.0], [1.0, 0.0]]), "numeric"),
        (torch.ones((2, 2)).to_sparse(), "dense"),
    ],
)
def test_fit_rejects_invalid_prior_masks(prior: object, match: str) -> None:
    with pytest.raises(HyperSCACError, match=match):
        fit_hypersca_c_once(
            [small_context()],
            small_config(maximum_epochs=1),
            seed=11,
            device="cpu",
            prior_mask=prior,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.5, "11"])
def test_fit_rejects_invalid_seeds(seed: object) -> None:
    with pytest.raises(HyperSCACError, match="seed"):
        fit_hypersca_c_once(
            [small_context()],
            small_config(maximum_epochs=1),
            seed=seed,  # type: ignore[arg-type]
            device="cpu",
        )


@pytest.mark.parametrize("device", ["not-a-device", "meta", 1])
def test_fit_wraps_invalid_devices_in_domain_error(device: object) -> None:
    with pytest.raises(HyperSCACError, match="device"):
        fit_hypersca_c_once(
            [small_context()],
            small_config(maximum_epochs=1),
            seed=11,
            device=device,  # type: ignore[arg-type]
        )


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_fit_rejects_unavailable_cuda_device() -> None:
    with pytest.raises(HyperSCACError, match="device"):
        fit_hypersca_c_once(
            [small_context()], small_config(maximum_epochs=1), seed=11, device="cuda"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fit_runs_on_cuda_and_returns_finite_cpu_arrays() -> None:
    result = fit_hypersca_c_once(
        [small_context()],
        small_config(maximum_epochs=2, early_stopping_patience=2),
        seed=11,
        device="cuda",
    )
    assert isinstance(result.shared, np.ndarray)
    assert np.isfinite(result.shared).all()
    assert np.isfinite(result.loss_history).all()
    assert all(
        isinstance(matrix, np.ndarray) and np.isfinite(matrix).all()
        for matrix in result.context_adjacencies.values()
    )


def test_fit_requires_unique_contexts_with_identical_ordered_genes() -> None:
    with pytest.raises(HyperSCACError, match="unique"):
        fit_hypersca_c_once(
            [small_context("same"), small_context("same")],
            small_config(maximum_epochs=1),
            seed=11,
            device="cpu",
        )
    with pytest.raises(HyperSCACError, match="same ordered genes"):
        fit_hypersca_c_once(
            [small_context("one"), small_context("two", gene_names=("B", "A"))],
            small_config(maximum_epochs=1),
            seed=11,
            device="cpu",
        )


def test_fit_requires_at_least_one_context_two_genes_and_two_controls() -> None:
    with pytest.raises(HyperSCACError, match="at least one context"):
        fit_hypersca_c_once([], small_config(), seed=11, device="cpu")
    one_gene = HyperSCACContext(
        context_id="one",
        expression=np.ones((2, 1)),
        interventions=np.asarray(["non-targeting"] * 2),
        gene_names=("A",),
    )
    with pytest.raises(HyperSCACError, match="at least two genes"):
        fit_hypersca_c_once([one_gene], small_config(), seed=11, device="cpu")
    one_control = HyperSCACContext(
        context_id="few-controls",
        expression=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        interventions=np.asarray(["non-targeting", "A"]),
        gene_names=("A", "B"),
    )
    with pytest.raises(HyperSCACError, match="two control"):
        fit_hypersca_c_once([one_control], small_config(), seed=11, device="cpu")


def _objective_for_fit(
    result: object,
    contexts: list[HyperSCACContext],
    config: HyperSCACConfig,
) -> float:
    fit = result
    shared = torch.as_tensor(fit.shared.copy())
    total = config.shared_l1 * shared.abs().mean()
    for context in contexts:
        values, _, _ = standardize_context(
            context.expression,
            context.interventions,
            control_label=config.control_label,
        )
        observed = torch.as_tensor(values)
        mask = torch.as_tensor(
            build_intervention_mask(
                context.interventions,
                context.gene_names,
                excluded_label=config.excluded_label,
            )
        )
        delta = torch.as_tensor(fit.context_adjustments[context.context_id].copy())
        adjacency = torch.as_tensor(
            fit.context_adjacencies[context.context_id].copy()
        )
        total = total + masked_sem_loss(observed @ adjacency, observed, mask)
        total = total + config.context_l1 * delta.abs().mean()
        total = total + config.acyclicity_weight * acyclicity_penalty(adjacency)
    return float(total)


def test_best_state_matches_the_best_recorded_loss() -> None:
    contexts = [small_context()]
    config = small_config(maximum_epochs=7, early_stopping_patience=7)
    result = fit_hypersca_c_once(contexts, config, seed=11, device="cpu")
    returned_loss = _objective_for_fit(result, contexts, config)
    assert returned_loss == pytest.approx(float(result.loss_history.min()), abs=2e-6)


def test_one_epoch_returns_the_first_updated_state_and_matching_loss() -> None:
    contexts = [small_context()]
    config = small_config(
        maximum_epochs=1,
        early_stopping_patience=1,
        enable_context_adjustments=False,
    )
    result = fit_hypersca_c_once(contexts, config, seed=11, device="cpu")
    assert np.any(result.shared != 0.0)
    returned_loss = _objective_for_fit(result, contexts, config)
    assert result.epochs_run == 1
    assert result.loss_history.tolist() == pytest.approx([returned_loss], abs=2e-6)


def test_best_state_keeps_even_a_small_recorded_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = HyperSCACContext(
        context_id="small-improvement",
        expression=np.asarray([[-1.0, -1.0], [1.0, 1.0], [2.0, 3.0], [2.0, 3.0]]),
        interventions=np.asarray(["non-targeting", "non-targeting", "A", "A"]),
        gene_names=("A", "B"),
    )

    def linear_loss(
        prediction: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return prediction.sum()

    def tiny_step(self: torch.optim.Adam, closure: object = None) -> None:
        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    parameter.add_(-1e-9)

    monkeypatch.setattr(hypersca_c_module, "masked_sem_loss", linear_loss)
    monkeypatch.setattr(torch.optim.Adam, "step", tiny_step)
    result = fit_hypersca_c_once(
        [context],
        small_config(
            maximum_epochs=2,
            early_stopping_patience=2,
            enable_context_adjustments=False,
            shared_l1=0.0,
            context_l1=0.0,
            acyclicity_weight=0.0,
        ),
        seed=11,
        device="cpu",
    )
    assert result.loss_history[1] < result.loss_history[0]
    assert result.shared[0, 1] != 0.0


def test_converged_only_means_patience_stopped_the_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    full = fit_hypersca_c_once(
        [small_context()],
        small_config(maximum_epochs=2, early_stopping_patience=5),
        seed=11,
        device="cpu",
    )
    assert full.epochs_run == 2
    assert full.converged is False

    def no_step(self: object, closure: object = None) -> None:
        return None

    monkeypatch.setattr(torch.optim.Adam, "step", no_step)
    stopped = fit_hypersca_c_once(
        [small_context()],
        small_config(maximum_epochs=8, early_stopping_patience=2),
        seed=11,
        device="cpu",
    )
    assert stopped.epochs_run == 3
    assert stopped.converged is True


def test_nonfinite_gradient_is_reported_as_domain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FiniteForwardNanGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
            ctx.input_shape = value.shape  # type: ignore[attr-defined]
            return value.sum() * 0.0

        @staticmethod
        def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
            return (
                torch.full(ctx.input_shape, float("nan")),  # type: ignore[attr-defined]
            )

    def broken_loss(
        prediction: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return FiniteForwardNanGradient.apply(prediction)

    monkeypatch.setattr(hypersca_c_module, "masked_sem_loss", broken_loss)
    with pytest.raises(HyperSCACError, match="gradient.*finite"):
        fit_hypersca_c_once(
            [small_context()],
            small_config(maximum_epochs=1),
            seed=11,
            device="cpu",
        )


def test_source_target_direction_uses_prediction_x_times_adjacency() -> None:
    rng = np.random.default_rng(11)
    rows = 120
    source = rng.normal(size=rows)
    target = 1.6 * source + rng.normal(scale=0.04, size=rows)
    unrelated = rng.normal(size=rows)
    context = HyperSCACContext(
        context_id="direction",
        expression=np.column_stack([source, target, unrelated]),
        interventions=np.asarray(["non-targeting"] * 20 + ["A"] * 100),
        gene_names=("A", "B", "C"),
    )
    result = fit_hypersca_c_once(
        [context],
        small_config(
            maximum_epochs=80,
            early_stopping_patience=20,
            enable_context_adjustments=False,
        ),
        seed=11,
        device="cpu",
    )
    # adjacency[source, target] and prediction = expression @ adjacency.
    assert abs(result.shared[0, 1]) > abs(result.shared[1, 0])
    assert abs(result.shared[0, 1]) > abs(result.shared[2, 1])


def valid_fit_payload() -> dict[str, object]:
    shared = np.asarray([[0.0, 0.2], [0.0, 0.0]], dtype=np.float64)
    delta = np.asarray([[0.0, 0.1], [0.0, 0.0]], dtype=np.float64)
    return {
        "shared": shared,
        "context_adjustments": {"k562": delta},
        "context_adjacencies": {"k562": shared + delta},
        "loss_history": np.asarray([1.0, 0.5]),
        "converged": False,
        "epochs_run": 2,
        "seed": 11,
        "config": small_config(maximum_epochs=2),
    }


def test_direct_fit_construction_normalizes_and_protects_valid_results() -> None:
    payload = valid_fit_payload()
    shared_input = payload["shared"]
    delta_input = payload["context_adjustments"]["k562"]  # type: ignore[index]
    adjacency_input = payload["context_adjacencies"]["k562"]  # type: ignore[index]
    history_input = payload["loss_history"]
    fit = HyperSCACFit(**payload)  # type: ignore[arg-type]
    shared_input[0, 1] = 9.0  # type: ignore[index]
    delta_input[0, 1] = 9.0  # type: ignore[index]
    adjacency_input[0, 1] = 9.0  # type: ignore[index]
    history_input[0] = 9.0  # type: ignore[index]
    assert fit.shared[0, 1] == pytest.approx(0.2)
    assert fit.context_adjustments["k562"][0, 1] == pytest.approx(0.1)
    assert fit.context_adjacencies["k562"][0, 1] == pytest.approx(0.3)
    assert fit.loss_history.tolist() == pytest.approx([1.0, 0.5])
    assert fit.shared.dtype == np.float32
    assert fit.loss_history.dtype == np.float64
    assert fit.config is payload["config"]
    assert fit.shared.flags.writeable is False
    assert fit.context_adjustments["k562"].flags.writeable is False
    assert fit.context_adjacencies["k562"].flags.writeable is False


def test_fit_arrays_cannot_have_write_permission_restored() -> None:
    fit = HyperSCACFit(**valid_fit_payload())  # type: ignore[arg-type]
    public_arrays = (
        fit.shared,
        fit.context_adjustments["k562"],
        fit.context_adjacencies["k562"],
        fit.loss_history,
    )
    for public_array in public_arrays:
        assert public_array.flags.writeable is False
        with pytest.raises(ValueError):
            public_array.setflags(write=True)


def test_fit_rejects_repeated_context_items_from_a_custom_mapping() -> None:
    class RepeatedItemsMapping(Mapping[str, np.ndarray]):
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def __getitem__(self, key: str) -> np.ndarray:
            if key != "k562":
                raise KeyError(key)
            return self.value

        def __iter__(self) -> Iterator[str]:
            return iter(("k562",))

        def __len__(self) -> int:
            return 1

        def items(self) -> tuple[tuple[str, np.ndarray], ...]:
            return (("k562", self.value), ("k562", self.value))

    payload = valid_fit_payload()
    repeated_delta = payload["context_adjustments"]["k562"]  # type: ignore[index]
    payload["context_adjustments"] = RepeatedItemsMapping(repeated_delta)
    with pytest.raises(HyperSCACError, match="context identifiers.*unique"):
        HyperSCACFit(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"shared": np.ones((2, 3))}, "shared.*square"),
        ({"shared": np.asarray([[0.0, np.nan], [0.0, 0.0]])}, "shared.*finite"),
        ({"shared": np.eye(2)}, "shared.*diagonal"),
        ({"shared": np.asarray([[1e-50, 0.2], [0.0, 0.0]])}, "shared.*diagonal"),
        ({"context_adjustments": {"bad id": np.zeros((2, 2))}}, "context.*whitespace"),
        ({"context_adjustments": {"k562": np.ones((2, 3))}}, "adjustment.*shape"),
        ({"context_adjustments": {"k562": np.eye(2)}}, "adjustment.*diagonal"),
        ({"context_adjacencies": {"rpe1": np.zeros((2, 2))}}, "same context"),
        ({"context_adjacencies": {"k562": np.zeros((2, 2))}}, "shared.*adjustment"),
        ({"loss_history": np.asarray([])}, "loss_history.*at least one"),
        ({"loss_history": np.asarray([1.0, np.inf])}, "loss_history.*finite"),
        ({"converged": 0}, "converged.*bool"),
        ({"epochs_run": 1}, "epochs_run.*history"),
        ({"epochs_run": True}, "epochs_run.*integer"),
        (
            {"loss_history": np.asarray([1.0, 0.5, 0.4]), "epochs_run": 3},
            "maximum_epochs",
        ),
        ({"seed": True}, "seed"),
        ({"config": object()}, "config"),
    ],
)
def test_direct_fit_construction_rejects_inconsistent_results(
    change: dict[str, object], match: str
) -> None:
    payload = valid_fit_payload()
    payload.update(change)
    with pytest.raises(HyperSCACError, match=match):
        HyperSCACFit(**payload)  # type: ignore[arg-type]
