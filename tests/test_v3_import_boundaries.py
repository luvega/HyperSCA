"""Executable Methods v3 architecture boundaries."""

from __future__ import annotations

import ast
import configparser
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".importlinter"


@dataclass(frozen=True)
class CliPolicy:
    functions: frozenset[str]
    calls: tuple[tuple[str, frozenset[str]], ...]
    assignments: tuple[tuple[str, frozenset[str]], ...]
    subscripts: tuple[tuple[str, frozenset[str]], ...]
    conditions: tuple[tuple[str, frozenset[str]], ...]
    returns: tuple[tuple[str, frozenset[str]], ...]
    attributes: tuple[tuple[str, frozenset[str]], ...]
    lazy_imports: frozenset[str]

    def calls_for(self, function_name: str) -> frozenset[str]:
        return dict(self.calls)[function_name]

    def assignments_for(self, function_name: str) -> frozenset[str]:
        return dict(self.assignments)[function_name]

    def subscripts_for(self, function_name: str) -> frozenset[str]:
        return dict(self.subscripts)[function_name]

    def conditions_for(self, function_name: str) -> frozenset[str]:
        return dict(self.conditions)[function_name]

    def returns_for(self, function_name: str) -> frozenset[str]:
        return dict(self.returns)[function_name]

    def attributes_for(self, function_name: str) -> frozenset[str]:
        return dict(self.attributes)[function_name]


PARSER_CALLS = frozenset(
    {"argparse.ArgumentParser", "parser.add_argument", "parser.parse_args"}
)
CLI_POLICIES = {
    "scripts/freeze_methods_protocol_outcome.py": CliPolicy(
        functions=frozenset({"parse_args", "main"}),
        calls=(
            ("parse_args", PARSER_CALLS),
            (
                "main",
                frozenset(
                    {
                        "parse_args",
                        "outcome_from_pilot_summary",
                        "write_protocol_outcome_exclusively",
                    }
                ),
            ),
        ),
        assignments=(
            (
                "parse_args",
                frozenset({"parser = argparse.ArgumentParser(description=__doc__)"}),
            ),
            ("main", frozenset({"args = parse_args()"})),
        ),
        subscripts=(("parse_args", frozenset()), ("main", frozenset())),
        conditions=(("parse_args", frozenset()), ("main", frozenset())),
        returns=(
            ("parse_args", frozenset({"parser.parse_args()"})),
            ("main", frozenset()),
        ),
        attributes=(
            (
                "parse_args",
                frozenset(
                    {
                        "argparse.ArgumentParser",
                        "argparse.Namespace",
                        "parser.add_argument",
                        "parser.parse_args",
                    }
                ),
            ),
            ("main", frozenset({"args.output", "args.pilot_summary"})),
        ),
        lazy_imports=frozenset(
            {
                "from src.evaluation.methods_protocol_outcome import "
                "outcome_from_pilot_summary, write_protocol_outcome_exclusively"
            }
        ),
    ),
    "scripts/audit_spatial_perturbation_bridge.py": CliPolicy(
        functions=frozenset({"_arguments", "main"}),
        calls=(
            (
                "_arguments",
                frozenset({"argparse.ArgumentParser", "parser.add_argument"}),
            ),
            (
                "main",
                frozenset(
                    {
                        "Path",
                        "Path().resolve",
                        "_arguments",
                        "_arguments().error",
                        "_arguments().parse_args",
                        "audit_bridge_capability",
                        "load_asset_metadata",
                        "load_bridge_candidates",
                        "str",
                        "sys.path.insert",
                        "unavailable_metadata_summary",
                        "write_bridge_capability_exclusively",
                    }
                ),
            ),
        ),
        assignments=(
            (
                "_arguments",
                frozenset(
                    {
                        "parser = argparse.ArgumentParser(description='audit "
                        "outcome-blind spatial perturbation capability')"
                    }
                ),
            ),
            (
                "main",
                frozenset(
                    {
                        "options = _arguments().parse_args()",
                        "repository_root = Path(__file__).resolve().parents[1]",
                        "registry_path = options.registry",
                        "repository_root_text = str(repository_root)",
                        "registry_path = repository_root / 'configs' / "
                        "'spatial_perturbation_bridge_candidates_v1.json'",
                        "candidates = load_bridge_candidates(registry_path)",
                        "candidate = candidates[options.candidate_id]",
                        "summary = load_asset_metadata(options.asset_root, "
                        "candidate) if options.asset_root is not None else "
                        "unavailable_metadata_summary(candidate)",
                    }
                ),
            ),
        ),
        subscripts=(
            ("_arguments", frozenset()),
            (
                "main",
                frozenset(
                    {
                        "Path(__file__).resolve().parents[1]",
                        "candidates[options.candidate_id]",
                    }
                ),
            ),
        ),
        conditions=(
            ("_arguments", frozenset()),
            (
                "main",
                frozenset(
                    {
                        "registry_path is None",
                        "repository_root_text not in sys.path",
                        "options.asset_root is not None",
                    }
                ),
            ),
        ),
        returns=(
            ("_arguments", frozenset({"parser"})),
            ("main", frozenset({"0"})),
        ),
        attributes=(
            (
                "_arguments",
                frozenset({"argparse.ArgumentParser", "parser.add_argument"}),
            ),
            (
                "main",
                frozenset(
                    {
                        "Path(__file__).resolve",
                        "Path(__file__).resolve().parents",
                        "_arguments().error",
                        "_arguments().parse_args",
                        "options.asset_root",
                        "options.candidate_id",
                        "options.output",
                        "options.registry",
                        "sys.path",
                        "sys.path.insert",
                    }
                ),
            ),
        ),
        lazy_imports=frozenset(
            {
                "from src.evaluation.spatial_perturbation_registry import "
                "SpatialPerturbationRegistryError, audit_bridge_capability, "
                "load_asset_metadata, load_bridge_candidates, "
                "unavailable_metadata_summary, write_bridge_capability_exclusively"
            }
        ),
    ),
    "scripts/validate_spatial_perturbation_predictor.py": CliPolicy(
        functions=frozenset({"_arguments", "main"}),
        calls=(
            (
                "_arguments",
                frozenset({"argparse.ArgumentParser", "parser.add_argument"}),
            ),
            (
                "main",
                frozenset(
                    {
                        "Path",
                        "Path().resolve",
                        "_arguments",
                        "audit_bridge_predictor_capability",
                        "parser.error",
                        "parser.parse_args",
                        "publish_spatial_perturbation_run",
                        "read_safe_declaration",
                        "str",
                        "sys.path.insert",
                    }
                ),
            ),
        ),
        assignments=(
            (
                "_arguments",
                frozenset(
                    {
                        "parser = argparse.ArgumentParser(allow_abbrev=False, "
                        "description='核查空间扰动预测方法是否已有预注册且可执行的接入层"
                        "（adapter），并发布不读取实验结局的证据边界记录。')"
                    }
                ),
            ),
            (
                "main",
                frozenset(
                    {
                        "parser = _arguments()",
                        "options = parser.parse_args()",
                        "repository_root = Path(__file__).resolve().parents[1]",
                        "repository_root_text = str(repository_root)",
                        "capability = audit_bridge_predictor_capability("
                        "read_safe_declaration(options.registry, label='预测方法注册表'), "
                        "read_safe_declaration(options.protocol, label='研究方案'), "
                        "method_id=options.method_id)",
                    }
                ),
            ),
        ),
        subscripts=(
            ("_arguments", frozenset()),
            (
                "main",
                frozenset({"Path(__file__).resolve().parents[1]"}),
            ),
        ),
        conditions=(
            ("_arguments", frozenset()),
            ("main", frozenset({"repository_root_text not in sys.path"})),
        ),
        returns=(
            ("_arguments", frozenset({"parser"})),
            ("main", frozenset({"0"})),
        ),
        attributes=(
            (
                "_arguments",
                frozenset({"argparse.ArgumentParser", "parser.add_argument"}),
            ),
            (
                "main",
                frozenset(
                    {
                        "Path(__file__).resolve",
                        "Path(__file__).resolve().parents",
                        "options.method_id",
                        "options.output_dir",
                        "options.protocol",
                        "options.registry",
                        "parser.error",
                        "parser.parse_args",
                        "sys.path",
                        "sys.path.insert",
                    }
                ),
            ),
        ),
        lazy_imports=frozenset(
            {
                "from src.evaluation.safe_declaration_reader import "
                "read_safe_declaration",
                "from src.evaluation.spatial_perturbation_predictor_contract "
                "import BridgePredictorContractError, "
                "audit_bridge_predictor_capability",
                "from src.evaluation.spatial_perturbation_runner import "
                "publish_spatial_perturbation_run",
            }
        ),
    ),
}

FROZEN_CONTRACT_IDS = frozenset(
    {
        "capability-audit-outcome-blind",
        "crc-promotion-isolated",
        "evidence-policy-domain-only",
        "predictor-contract-model-free",
        "publisher-model-free",
        "scoring-policy-free",
        "split-model-free",
    }
)
LEGACY_RUN_EVIDENCE_CONTRACT_IDS = frozenset(
    {
        "run-evidence-layers",
        "run-evidence-no-benchmark-runners",
        "run-evidence-no-causal",
        "run-evidence-no-data",
        "run-evidence-no-discovery",
        "run-evidence-no-models",
        "run-evidence-no-orchestration",
        "run-evidence-no-perturbation",
    }
)
NEW_CONTRACTS = {
    "evidence-policy-domain-only": {
        "type": ("forbidden",),
        "source_modules": ("src.discovery.evidence_policy",),
        "forbidden_modules": (
            "src.models",
            "src.causal",
            "src.perturbation",
            "src.data",
            "src.evaluation.methods_pilot",
            "src.evaluation.methods_causal_pilot",
            "src.evaluation.task_s_benchmark",
            "src.evaluation.task_c_benchmark",
            "src.evaluation.task_c_method_run",
            "src.evaluation.task_c_aggregation",
            "src.evaluation.task_c_rehearsal",
            "src.evaluation.task_c_runtime",
            "src.evaluation.task_c_data",
            "src.evaluation.task_c_acquisition",
            "src.evaluation.task_c_profile_input",
            "src.evaluation.task_c_formal_export",
            "src.evaluation.task_c_predictions",
            "src.evaluation.task_c_tuning",
            "src.evaluation.safe_declaration_reader",
            "src.evaluation.spatial_perturbation_runner",
        ),
    },
    "publisher-model-free": {
        "type": ("forbidden",),
        "source_modules": ("src.evaluation.run_evidence_publisher",),
        "forbidden_modules": ("src.models", "src.causal", "src.perturbation"),
    },
    "capability-audit-outcome-blind": {
        "type": ("forbidden",),
        "source_modules": ("src.evaluation.spatial_perturbation_registry",),
        "forbidden_modules": (
            "src.models",
            "src.causal",
            "src.perturbation",
            "src.data",
            "src.discovery",
            "src.evaluation.msi_inference",
            "src.evaluation.benchmark_evidence",
            "src.evaluation.spatial_perturbation_scoring",
            "src.evaluation.causal_metrics",
            "src.evaluation.cf_metrics",
            "src.evaluation.cross_sample_metrics",
            "src.evaluation.embedding_metrics",
            "src.evaluation.spatial_metrics",
            "src.evaluation.spatial_perturbation_runner",
            "src.evaluation.task_s_benchmark",
            "src.evaluation.task_c_benchmark",
            "src.evaluation.task_c_aggregation",
            "src.evaluation.task_c_tuning",
            "src.evaluation.task_c_null_controls",
            "src.evaluation.task_c_profile_input",
            "src.evaluation.task_c_data",
            "src.evaluation.task_c_acquisition",
            "src.evaluation.task_c_formal_export",
            "src.evaluation.task_c_predictions",
            "src.evaluation.task_c_runtime",
            "src.evaluation.task_c_rehearsal",
            "src.evaluation.task_c_method_registry",
            "src.evaluation.task_c_method_run",
            "src.evaluation.methods_protocol_outcome",
            "src.evaluation.methods_pilot",
            "src.evaluation.methods_causal_pilot",
        ),
    },
    "split-model-free": {
        "type": ("forbidden",),
        "source_modules": ("src.evaluation.spatial_perturbation_split",),
        "forbidden_modules": (
            "src.models",
            "src.discovery.evidence_policy",
        ),
    },
    "scoring-policy-free": {
        "type": ("forbidden",),
        "source_modules": ("src.evaluation.spatial_perturbation_scoring",),
        "forbidden_modules": (
            "src.discovery.evidence_policy",
            "src.evaluation.run_evidence_publisher",
        ),
    },
    "predictor-contract-model-free": {
        "type": ("forbidden",),
        "source_modules": ("src.evaluation.spatial_perturbation_predictor_contract",),
        "forbidden_modules": ("src.models", "src.causal", "src.perturbation"),
    },
    "crc-promotion-isolated": {
        "type": ("forbidden",),
        "source_modules": (
            "src.discovery.from_scratch.crc_icb_inputs",
            "src.discovery.from_scratch.crc_icb_artifacts",
        ),
        "forbidden_modules": (
            "src.discovery.evidence_policy",
            "src.evaluation.methods_protocol_v3",
        ),
    },
}

MODULE_IMPORTS = frozenset(
    {
        "from __future__ import annotations",
        "import argparse",
        "from pathlib import Path",
        "import sys",
    }
)
ALLOWED_FUNCTION_NODES = frozenset(
    {
        ast.Add,
        ast.Assign,
        ast.Attribute,
        ast.BinOp,
        ast.Call,
        ast.Compare,
        ast.Constant,
        ast.Div,
        ast.ExceptHandler,
        ast.Expr,
        ast.If,
        ast.IfExp,
        ast.ImportFrom,
        ast.Is,
        ast.IsNot,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.List,
        ast.Load,
        ast.Name,
        ast.NotIn,
        ast.Return,
        ast.Store,
        ast.Subscript,
        ast.Try,
        ast.Tuple,
        ast.alias,
        ast.arg,
        ast.arguments,
        ast.keyword,
    }
)
PREDICTOR_CONTRACT_IMPORTS = frozenset(
    {
        "from __future__ import annotations",
        "from dataclasses import dataclass",
        "import hashlib",
        "import json",
        "import math",
        "import re",
        "from types import MappingProxyType",
        "from typing import Any, Mapping, cast",
        "import unicodedata",
        "from src.evaluation.run_evidence_identity import MAX_EXACT_INTEGER, "
        "RunEvidenceError, canonical_json_bytes, canonical_sha256, "
        "validate_strict_json",
        "from src.methods_protocol_v3_contract import MethodsProtocolV3, "
        "build_methods_protocol_v3, protocol_identity_v3, "
        "protocol_to_mapping_v3",
    }
)
# Review workflow: a deliberate predictor-contract edit must identify the changed
# top-level declaration here.  The per-declaration manifest keeps review diffs
# local while the recursive digest covers every semantic AST field beneath it.
PREDICTOR_CONTRACT_DECLARATION_GOLDEN = {
    "<module declarations>": "5d42751fa4ce225d365e3bf13309aa93441b6305de4183f1209ca49886e40141",
    "function _fresh_prediction_schema": "9b8b562b70e33876a2eb803bb55e1fe082d5936ecdde007ee78cb937fb57097e",
    "class BridgePredictorContractError": "ed7f55884f99b8957eef58180cdf6e232b1b08f7a5daec824e282fb4b4185ccf",
    "function _fail": "7225194bb6b5b25a720005483e70897d525b90613582094f60897c49def36473",
    "function _safe_text": "987471ff52b7ca078ecff3565b5cfacf29778106f39ef555e2daf45209d12724",
    "function _sha": "499bc346cbefc56461b3d8a888c9c8ec6f03240b16cb4a720f61fdb5f45e2790",
    "function _seed": "632171099cf806bb98d3935dbdc2350c570e7dba9a9e97aa5c43d012d007ede5",
    "function _strict_tree": "837ae5d272287544b1db8c896c6a416693fbe666bfcc975444668f2ed7d3e536",
    "function _reject_unsafe_declaration_keys": "480c33f99b7c7c7b503443fce89f7ae10c4fedf83859acbe8c3767a995c3b4d5",
    "function formal_bridge_predictor_registry_declaration_to_mapping": "1549869856e9c1d204aca3a7197ef71d96699b26819075ae9c51bdd5a0bd2bba",
    "function formal_protocol_declaration_to_mapping": "08c85dc7836f8268ce292d224540cea2a5b9dd2576dc74c4be0c7c342afa642e",
    "function formal_protocol_declaration_identity_sha256": "50df750c6681361776fa5246c47a4e52a9fa71e76026b75253615c7336c7be6c",
    "function _capability_unsigned": "2b68fa3ff264ce81eaff2ad681707c38584e4430735237a4e51741f0d28204e9",
    "class BridgePredictorCapability": "93248dde114b05df83b7e0aeba82e6404650e9b86cb0a632a995bf55ebdfc0b4",
    "function _snapshot_capability": "681ebb79f9f21a6ed1e5bfacdbb76114235b010adebdd89a5d7b53484ff2ed24",
    "function audit_bridge_predictor_capability": "66e66bb9176f522d2adffc897b0b268a0d7fe57d53d6f1be50408fb8d3cf9bb1",
    "function bridge_predictor_capability_to_mapping": "ce7bf7413f8da453bc480c23bb1ebaea911778bc3c5fb6048c7f4ec95777a2d9",
    "function _deep_freeze": "4b49267c5882f93e9e4446eba8b4224bb9f01a18c6a9b69aeaa70520883bd45f",
    "function _strict_json_bytes": "75cf1f65b79d050abd5df856a0ea1a48f3b8faa9b94ae433f003285b59ab97c3",
    "function _validate_schema": "638145e0f0de1526ef4306db70c9d4bdbe79ba37a1e97cfbc454c3232d258bd6",
    "function _schema_record": "daf2c9a74a06fbb685faedf0b852a93d63df9dba71a7565651cb5699b9ef8e6c",
    "function _validate_prediction_payload": "5c6e91f2177b5bb26bacc7aaf6cd13188a086473da3820786ba0d773053635a2",
    "function _bundle_unsigned": "6bc84ae6f514f6bf938dc5be7f41f1cd3fee537cf31b96421666c02f97b157e7",
    "class BridgePredictionBundle": "6a3fb55d87ae7aa1254fe4773f52decd0b8013d7b3151b7727361fb89ad81389",
    "function _snapshot_bundle": "3387f1406f5659c97be9bbd70bb7eea123ab472bc8f12eb917decb6f90d25e50",
    "function build_bridge_prediction_bundle": "651cfc7c5fb1b22a179600b5cc77805e61a79e232cc5663b0279a6a660e6d3af",
    "function bridge_prediction_bundle_to_mapping": "3b2c3a9728298cc830bcf1b8a0b14c634588e7408923983310764bfb9b5086c9",
    "function prediction_payload_to_mapping": "b2e5c718c3164d51a64a5a955f61fa8b43c2f05f0972585843c4d7630bcb5e74",
}


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_name_guard(node: ast.If) -> bool:
    return ast.unparse(node.test) == "__name__ == '__main__'"


def _is_valid_name_guard(node: ast.If) -> bool:
    if not _is_name_guard(node) or node.orelse or len(node.body) != 1:
        return False
    statement = node.body[0]
    if isinstance(statement, ast.Expr):
        return ast.unparse(statement.value) == "main()"
    return (
        isinstance(statement, ast.Raise)
        and statement.exc is not None
        and ast.unparse(statement.exc) == "SystemExit(main())"
    )


def _is_path_bootstrap(node: ast.If) -> bool:
    return (
        ast.unparse(node.test) == "str(ROOT) not in sys.path"
        and not node.orelse
        and len(node.body) == 1
        and ast.unparse(node.body[0]) == "sys.path.insert(0, str(ROOT))"
    )


def _call_signature(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_signature(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return f"{_call_signature(node.func)}()"
    if isinstance(node, ast.Subscript):
        return f"{_call_signature(node.value)}[]"
    return f"<{type(node).__name__}>"


def _is_string_expression(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)) or (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and _is_string_expression(node.left)
        and _is_string_expression(node.right)
    )


def _is_error_string_expression(node: ast.AST) -> bool:
    if _is_string_expression(node):
        return True
    if isinstance(node, ast.Call):
        return (
            _call_signature(node.func) == "str"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in {"error", "exc"}
            and not node.keywords
        )
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and _is_error_string_expression(node.left)
        and _is_error_string_expression(node.right)
    )


def _is_path_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "repository_root"
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and _is_path_expression(node.left)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, str)
    )


def _parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(root)
        for child in ast.iter_child_nodes(parent)
    }


def _is_error_translation_context(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.Call) and _call_signature(current.func) in {
            "_arguments().error",
            "parser.error",
        }:
            return True
        if isinstance(current, ast.stmt):
            return False
    return False


def _import_bindings(rendered_imports: frozenset[str]) -> frozenset[str]:
    bindings = {"Path", "argparse", "parser", "str", "sys"}
    for rendered in rendered_imports:
        statement = ast.parse(rendered).body[0]
        if isinstance(statement, ast.Import):
            bindings.update(
                alias.asname or alias.name.split(".", 1)[0] for alias in statement.names
            )
        elif isinstance(statement, ast.ImportFrom):
            bindings.update(alias.asname or alias.name for alias in statement.names)
    return frozenset(bindings)


def _function_violations(
    function: ast.FunctionDef,
    *,
    allowed_calls: frozenset[str],
    allowed_assignments: frozenset[str],
    allowed_subscripts: frozenset[str],
    allowed_conditions: frozenset[str],
    allowed_returns: frozenset[str],
    allowed_attributes: frozenset[str],
    allowed_lazy_imports: frozenset[str],
) -> list[str]:
    violations: list[str] = []
    parents = _parent_map(function)
    protected_bindings = _import_bindings(allowed_lazy_imports)
    if function.decorator_list:
        violations.append(f"decorator on {function.name}")
    if function.args.defaults or any(
        default is not None for default in function.args.kw_defaults
    ):
        violations.append(f"default argument on {function.name}")
    if (
        function.args.posonlyargs
        or function.args.args
        or function.args.kwonlyargs
        or function.args.vararg is not None
        or function.args.kwarg is not None
    ):
        violations.append(f"arguments on {function.name}")

    for node in ast.walk(function):
        if node is function:
            continue
        if isinstance(node, ast.ExceptHandler) and node.name in protected_bindings:
            violations.append(f"forbidden binding {node.name}")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            violations.append(f"nested helper in {function.name}")
            continue
        if isinstance(node, ast.Expr):
            if not _is_docstring(node) and not isinstance(node.value, ast.Call):
                violations.append(f"forbidden expression {ast.unparse(node.value)}")
            continue
        if isinstance(node, ast.Return):
            rendered_return = "None" if node.value is None else ast.unparse(node.value)
            if rendered_return not in allowed_returns:
                violations.append(f"forbidden return {rendered_return}")
            continue
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, (ast.Attribute, ast.Subscript))
                for target in node.targets
            ):
                violations.append(
                    f"forbidden assignment target {ast.unparse(node.targets[0])}"
                )
            if ast.unparse(node) not in allowed_assignments:
                violations.append(f"forbidden assignment {ast.unparse(node)}")
            continue
        if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            target = node.target
            if isinstance(target, (ast.Attribute, ast.Subscript)):
                violations.append(f"forbidden assignment target {ast.unparse(target)}")
            violations.append(f"forbidden assignment {ast.unparse(node)}")
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rendered = ast.unparse(node)
            if function.name != "main" or rendered not in allowed_lazy_imports:
                violations.append(f"forbidden import {rendered}")
            continue
        if isinstance(node, ast.Call):
            signature = _call_signature(node.func)
            if signature not in allowed_calls:
                violations.append(f"forbidden call {signature}")
            elif signature == "str":
                rendered = ast.unparse(node)
                if rendered == "str(repository_root)":
                    pass
                elif rendered not in {"str(error)", "str(exc)"} or not (
                    _is_error_translation_context(node, parents)
                ):
                    violations.append("forbidden call str outside error translation")
            continue
        if isinstance(node, ast.Attribute):
            if ast.unparse(node) not in allowed_attributes:
                violations.append(f"forbidden attribute {ast.unparse(node)}")
            continue
        if isinstance(node, ast.Subscript):
            if ast.unparse(node) not in allowed_subscripts:
                violations.append(f"forbidden subscript {ast.unparse(node)}")
            continue
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) not in allowed_conditions
        ):
            violations.append(f"forbidden branch {ast.unparse(node.test)}")
        elif (
            isinstance(node, ast.IfExp)
            and ast.unparse(node.test) not in allowed_conditions
        ):
            violations.append(f"forbidden branch {ast.unparse(node.test)}")
        elif isinstance(node, ast.BinOp):
            allowed_binop = _is_path_expression(node) or (
                _is_error_string_expression(node)
                and _is_error_translation_context(node, parents)
            )
            if not allowed_binop:
                violations.append(f"forbidden calculation {ast.unparse(node)}")
        elif isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            if not _is_error_translation_context(node, parents):
                violations.append(
                    "forbidden formatted string outside error translation"
                )
        if type(node) not in ALLOWED_FUNCTION_NODES:
            violations.append(f"forbidden structure {type(node).__name__}")
    return violations


def _thin_cli_violations(
    source: str,
    *,
    policy: CliPolicy,
) -> tuple[str, ...]:
    tree = ast.parse(source)
    violations: list[str] = []
    functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if _is_docstring(node):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rendered = ast.unparse(node)
            if rendered not in MODULE_IMPORTS:
                violations.append(f"forbidden module import {rendered}")
            continue
        if isinstance(node, ast.Assign):
            if ast.unparse(node) != "ROOT = Path(__file__).resolve().parents[1]":
                violations.append(f"forbidden top-level Assign {ast.unparse(node)}")
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name not in policy.functions:
                violations.append(f"unexpected top-level function {node.name}")
            elif node.name in functions:
                violations.append(f"duplicate CLI function {node.name}")
            else:
                functions[node.name] = node
            continue
        if isinstance(node, ast.If):
            if not (_is_valid_name_guard(node) or _is_path_bootstrap(node)):
                violations.append(f"forbidden module branch {ast.unparse(node.test)}")
            continue
        violations.append(f"forbidden top-level {type(node).__name__}")

    if set(functions) != set(policy.functions):
        violations.append("missing required CLI function")
    for function_name, function in functions.items():
        violations.extend(
            _function_violations(
                function,
                allowed_calls=policy.calls_for(function_name),
                allowed_assignments=policy.assignments_for(function_name),
                allowed_subscripts=policy.subscripts_for(function_name),
                allowed_conditions=policy.conditions_for(function_name),
                allowed_returns=policy.returns_for(function_name),
                allowed_attributes=policy.attributes_for(function_name),
                allowed_lazy_imports=policy.lazy_imports,
            )
        )
    return tuple(dict.fromkeys(violations))


def _generic_policy(functions: frozenset[str]) -> CliPolicy:
    return CliPolicy(
        functions=functions,
        calls=tuple(
            (
                function,
                frozenset({"argparse.ArgumentParser", "parser.error", "str"}),
            )
            for function in sorted(functions)
        ),
        assignments=tuple((function, frozenset()) for function in sorted(functions)),
        subscripts=tuple((function, frozenset()) for function in sorted(functions)),
        conditions=tuple((function, frozenset()) for function in sorted(functions)),
        returns=tuple(
            (function, frozenset({"None"})) for function in sorted(functions)
        ),
        attributes=tuple((function, frozenset()) for function in sorted(functions)),
        lazy_imports=frozenset(),
    )


def _canonical_ast_value(value: object) -> object:
    if isinstance(value, ast.AST):
        fields = []
        for field in value._fields:
            field_value = getattr(value, field)
            # Python 3.12 added empty ``type_params`` to definitions.  Empty
            # version-only metadata is non-semantic; a non-empty value remains
            # part of the digest.  Location metadata is explicitly ignored.
            if field in {
                "lineno",
                "col_offset",
                "end_lineno",
                "end_col_offset",
            } and not (isinstance(value, ast.TypeIgnore) and field == "lineno"):
                continue
            if field == "type_params" and not field_value:
                continue
            fields.append((field, _canonical_ast_value(field_value)))
        return (
            type(value).__name__,
            tuple(fields),
        )
    if isinstance(value, list):
        return tuple(_canonical_ast_value(item) for item in value)
    return (type(value).__name__, repr(value))


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _canonical_ast_digest(value: ast.AST) -> str:
    return _canonical_digest(_canonical_ast_value(value))


def _predictor_contract_declaration_manifest(
    tree: ast.Module,
) -> tuple[dict[str, str], tuple[str, ...]]:
    declarations = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    module_layout: list[object] = [
        ("module field type_ignores", _canonical_ast_value(tree.type_ignores))
    ]
    manifest: dict[str, str] = {}
    duplicates: list[str] = []
    for node in tree.body:
        if isinstance(node, declarations):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            key = f"{kind} {node.name}"
            module_layout.append(("declaration", key))
            if key in manifest:
                duplicates.append(key)
            manifest[key] = _canonical_ast_digest(node)
        else:
            module_layout.append(("statement", _canonical_ast_value(node)))
    return {
        "<module declarations>": _canonical_digest(tuple(module_layout)),
        **manifest,
    }, tuple(duplicates)


def _predictor_contract_model_violations(source: str) -> tuple[str, ...]:
    tree = ast.parse(source, type_comments=True)
    violations: list[str] = []
    rendered_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rendered = ast.unparse(node)
            rendered_imports.append(rendered)
            if rendered not in PREDICTOR_CONTRACT_IMPORTS:
                violations.append(f"forbidden predictor import {rendered}")
    if frozenset(rendered_imports) != PREDICTOR_CONTRACT_IMPORTS or len(
        rendered_imports
    ) != len(PREDICTOR_CONTRACT_IMPORTS):
        violations.append("predictor imports differ from exact allowlist")
    actual_manifest, duplicates = _predictor_contract_declaration_manifest(tree)
    for duplicate in duplicates:
        violations.append(f"duplicate predictor declaration {duplicate}")
    expected_keys = set(PREDICTOR_CONTRACT_DECLARATION_GOLDEN)
    actual_keys = set(actual_manifest)
    for missing in sorted(expected_keys - actual_keys):
        violations.append(f"missing predictor declaration {missing}")
    for unexpected in sorted(actual_keys - expected_keys):
        violations.append(f"unexpected predictor declaration {unexpected}")
    for key in sorted(expected_keys & actual_keys):
        if actual_manifest[key] != PREDICTOR_CONTRACT_DECLARATION_GOLDEN[key]:
            violations.append(f"changed predictor declaration {key}")
    return tuple(dict.fromkeys(violations))


def test_import_linter_contracts_pass() -> None:
    result = subprocess.run(
        ["lint-imports", "--config", ".importlinter"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_exact_new_and_existing_import_linter_contracts_are_configured() -> None:
    parser = configparser.ConfigParser()
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        parser.read_file(handle)
    contract_ids = {
        section.removeprefix("importlinter:contract:")
        for section in parser.sections()
        if section.startswith("importlinter:contract:")
    }
    assert contract_ids == FROZEN_CONTRACT_IDS | LEGACY_RUN_EVIDENCE_CONTRACT_IDS
    for contract_id, expected in NEW_CONTRACTS.items():
        section = f"importlinter:contract:{contract_id}"
        for option, expected_values in expected.items():
            assert tuple(parser.get(section, option).split()) == expected_values


def test_development_extra_declares_architecture_test_dependencies() -> None:
    try:
        tomllib = importlib.import_module("tomllib")
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
        tomllib = importlib.import_module("tomli")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["optional-dependencies"]["dev"] == [
        "pytest>=7,<9",
        "tomli>=2; python_version < '3.11'",
        "hypothesis>=6,<7",
        "import-linter>=2,<3",
    ]


def test_plan_defines_seven_import_contracts_plus_one_cli_boundary() -> None:
    plan = (
        ROOT / "docs/superpowers/plans/2026-08-28-hypersca-methods-v3-bridge.md"
    ).read_text(encoding="utf-8")
    task = plan.split("## Task 10:", 1)[1].split("## Task 11:", 1)[0]
    assert "[importlinter:contract:runner-crc-isolated]" not in task
    assert task.count("[importlinter:contract:") == 7
    assert "seven Import Linter contracts plus one AST contract" in task


@pytest.mark.parametrize("relative_path", tuple(CLI_POLICIES))
def test_methods_v3_clis_remain_thin(relative_path: str) -> None:
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    assert (
        _thin_cli_violations(
            source,
            policy=CLI_POLICIES[relative_path],
        )
        == ()
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("from src.models import HyperSCA\n", "forbidden module import"),
        (
            "def main():\n    import numpy as np\n    return np.array([1, 2])\n",
            "forbidden import",
        ),
        ("def main():\n    model = SpatialModel()\n", "forbidden call"),
        (
            "def main():\n    effect = treated.mean() - control.mean()\n",
            "forbidden call",
        ),
        (
            "def compute_metric():\n    return 1\ndef main():\n    return compute_metric()\n",
            "unexpected top-level function",
        ),
        (
            "import importlib\ndef main():\n"
            "    return importlib.import_module('src.causal.engine')\n",
            "forbidden module import",
        ),
        (
            "def main():\n"
            "    from src.models import HyperSCA\n"
            "    return HyperSCA()\n",
            "forbidden import",
        ),
        (
            "CACHE = load_outcomes()\ndef main():\n    return CACHE\n",
            "forbidden top-level Assign",
        ),
        (
            "import statistics\ndef main():\n"
            "    return statistics.fmean([1.0, 2.0])\n",
            "forbidden call",
        ),
        ("def main():\n    return sum([1, 2])\n", "forbidden call"),
        (
            "def main():\n    values = [2, 1]\n    values.sort()\n",
            "forbidden call",
        ),
        (
            "def main():\n    if treatment > control:\n        return treatment\n",
            "forbidden branch",
        ),
        ("@decorate\ndef main():\n    return None\n", "decorator"),
        (
            "def main(value=load_outcomes()):\n    return value\n",
            "default argument",
        ),
        (
            "bootstrap()\ndef main():\n    return None\n",
            "forbidden top-level Expr",
        ),
        (
            "import importlib as loader\ndef main():\n"
            "    module_name = 'src.' + 'models'\n"
            "    return loader.import_module(module_name)\n",
            "forbidden module import",
        ),
        (
            "def main():\n    return __import__('numpy')\n",
            "forbidden call",
        ),
        (
            "def main():\n    import numpy as np\n"
            "    return getattr(np, 'array')([1])\n",
            "forbidden call",
        ),
        (
            "def main():\n    return factory.SpatialModel()\n",
            "forbidden call",
        ),
        (
            "def main():\n    def helper():\n        return sum([1, 2])\n"
            "    return helper()\n",
            "nested helper",
        ),
        (
            "def main():\n    return None\n" "def main():\n    return sum([1, 2])\n",
            "duplicate CLI function",
        ),
        (
            "def main():\n    str = eval\n    return str('payload')\n",
            "forbidden assignment",
        ),
        (
            "def main():\n    parser.error = eval\n"
            "    return parser.error('payload')\n",
            "forbidden assignment target",
        ),
        (
            "def main():\n    try:\n        return None\n"
            "    except Exception as str:\n        return parser.error(str(exc))\n",
            "forbidden binding",
        ),
        ("def main():\n    return data[0]\n", "forbidden subscript"),
        (
            "def main():\n    capability.predictions\n",
            "forbidden expression",
        ),
        ("def main():\n    return frame.T\n", "forbidden return"),
    ),
)
def test_cli_ast_contract_rejects_adversarial_scientific_mutations(
    mutation: str,
    expected: str,
) -> None:
    violations = _thin_cli_violations(
        mutation,
        policy=_generic_policy(frozenset({"main"})),
    )
    assert any(expected in violation for violation in violations), violations


def test_cli_ast_contract_allows_error_message_string_concatenation() -> None:
    path = "scripts/validate_spatial_perturbation_predictor.py"
    source = (
        (ROOT / path)
        .read_text(encoding="utf-8")
        .replace(
            "parser.error(str(error))",
            "parser.error('error: ' + str(error))",
        )
    )
    assert _thin_cli_violations(source, policy=CLI_POLICIES[path]) == ()


def test_cli_ast_contract_allows_error_translation_f_string() -> None:
    path = "scripts/validate_spatial_perturbation_predictor.py"
    source = (
        (ROOT / path)
        .read_text(encoding="utf-8")
        .replace(
            "parser.error(str(error))",
            'parser.error(f"error: {error}")',
        )
    )
    assert _thin_cli_violations(source, policy=CLI_POLICIES[path]) == ()


def test_predictor_contract_source_cannot_import_or_fit_external_models() -> None:
    source = (
        ROOT / "src/evaluation/spatial_perturbation_predictor_contract.py"
    ).read_text(encoding="utf-8")
    assert _predictor_contract_model_violations(source) == ()


@pytest.mark.parametrize(
    "mutation",
    (
        "import torch\n",
        "from sklearn.linear_model import Ridge\n",
        "def probe(model):\n    return model.fit([[1.0]], [1.0])\n",
        "def probe():\n    return __import__('torch')\n",
        "import importlib\ndef probe():\n"
        "    return importlib.import_module('sklearn')\n",
        "import importlib as loader\ndef probe():\n"
        "    return loader.import_module('torch')\n",
        "import tensorflow\n",
        "load = __import__\n",
        "def probe(payload):\n    return exec(payload)\n",
        "def probe():\n    return getattr(__builtins__, '__import__')\n",
        "def probe(model):\n    fit = model.fit\n    return fit()\n",
        "def probe(model):\n    optimize = model.optimize\n    return optimize()\n",
        "def probe(model):\n    return model.__getattribute__('fit')\n",
        "def probe():\n    return globals()['__builtins__']\n",
    ),
)
def test_predictor_contract_ast_rejects_external_model_escape_hatches(
    mutation: str,
) -> None:
    source = (
        ROOT / "src/evaluation/spatial_perturbation_predictor_contract.py"
    ).read_text(encoding="utf-8")
    assert _predictor_contract_model_violations(f"{source}\n{mutation}")


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "\n__all__ = [",
            "\ndef hidden_matmul(left, right):\n    return left @ right\n\n__all__ = [",
        ),
        (
            "\n__all__ = [",
            "\ndef fabricated_result():\n"
            "    return {'status': 'completed'}\n\n__all__ = [",
        ),
        ("type(value) is not str", "type(value) is str"),
        (
            '_FIXED_STATUS = "method_adapter_not_executable"',
            '_FIXED_STATUS = "completed"',
        ),
        ("    return normalized\n", "    return model\n"),
        (
            "\n__all__ = [",
            "\ndef raise_only():\n    raise\n\n__all__ = [",
        ),
    ),
)
def test_predictor_contract_ast_rejects_any_semantic_declaration_change(
    old: str,
    new: str,
) -> None:
    source = (
        ROOT / "src/evaluation/spatial_perturbation_predictor_contract.py"
    ).read_text(encoding="utf-8")
    assert old in source
    mutated = source.replace(old, new, 1)
    assert _predictor_contract_model_violations(mutated)


def test_predictor_contract_manifest_names_the_changed_declaration() -> None:
    source = (
        ROOT / "src/evaluation/spatial_perturbation_predictor_contract.py"
    ).read_text(encoding="utf-8")
    changed_function = source.replace(
        "type(value) is not str",
        "type(value) is str",
        1,
    )
    assert _predictor_contract_model_violations(changed_function) == (
        "changed predictor declaration function _safe_text",
    )

    added_function = source.replace(
        "\n__all__ = [",
        "\ndef hidden_matmul(left, right):\n    return left @ right\n\n__all__ = [",
        1,
    )
    violations = _predictor_contract_model_violations(added_function)
    assert "unexpected predictor declaration function hidden_matmul" in violations
    assert "changed predictor declaration <module declarations>" in violations


def test_predictor_contract_manifest_binds_type_ignore_to_its_line() -> None:
    source = (
        ROOT / "src/evaluation/spatial_perturbation_predictor_contract.py"
    ).read_text(encoding="utf-8")
    on_hashlib = source.replace(
        "import hashlib",
        "import hashlib  # type: ignore[import-not-found]",
        1,
    )
    on_json = source.replace(
        "import json",
        "import json  # type: ignore[import-not-found]",
        1,
    )
    hashlib_manifest, _ = _predictor_contract_declaration_manifest(
        ast.parse(on_hashlib, type_comments=True)
    )
    json_manifest, _ = _predictor_contract_declaration_manifest(
        ast.parse(on_json, type_comments=True)
    )
    assert (
        hashlib_manifest["<module declarations>"]
        != json_manifest["<module declarations>"]
    )
