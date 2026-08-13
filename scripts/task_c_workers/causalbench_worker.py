#!/usr/bin/env python3
"""Run one fixed CausalBench method on an already permitted cell subset."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import csv
import importlib
import io
import os
from pathlib import Path
import secrets
import site
import stat
import subprocess
import sys
from typing import Any, Sequence
import unicodedata
import zipfile


MODEL_NAMES = (
    "random1000",
    "grnboost",
    "pc",
    "ges",
    "gies",
    "gsp",
    "igsp",
    "notears-lin-sparse",
    "DCDI-G",
    "DCDI-DSF",
    "DCDFG-LIN",
    "DCDFG-MLP",
    "sortnregress",
)
MODEL_OUTPUT_SEMANTICS = {
    "random1000": "official_unranked_edges",
    "grnboost": "official_return_order",
    "pc": "official_unranked_edges",
    "ges": "official_unranked_edges",
    "gies": "official_unranked_edges",
    "gsp": "official_unranked_edges",
    "igsp": "official_unranked_edges",
    "notears-lin-sparse": "official_unranked_edges",
    "DCDI-G": "official_unranked_edges",
    "DCDI-DSF": "official_unranked_edges",
    "DCDFG-LIN": "official_unranked_edges",
    "DCDFG-MLP": "official_unranked_edges",
    "sortnregress": "official_unranked_edges",
}
"""Fixed evidence boundary for ranked versus unranked official results."""
MODEL_TRAINING_INFORMATION = {
    "random1000": "observational",
    "grnboost": "observational",
    "pc": "observational",
    "ges": "observational",
    "gies": "partial_interventional",
    "gsp": "observational",
    "igsp": "partial_interventional",
    "notears-lin-sparse": "observational",
    "DCDI-G": "partial_interventional",
    "DCDI-DSF": "partial_interventional",
    "DCDFG-LIN": "partial_interventional",
    "DCDFG-MLP": "partial_interventional",
    "sortnregress": "observational",
}
MODEL_IMPORTS = {
    "random1000": (
        "causalscbench.models.random_network",
        "RandomWithSize",
        (1000,),
        {},
    ),
    "grnboost": ("causalscbench.models.arboreto_baselines", "GRNBoost", (), {}),
    "pc": ("causalscbench.models.causallearn_models", "PC", (), {"missing_value": False}),
    "ges": ("causalscbench.models.causallearn_models", "GES", (), {}),
    "gies": ("causalscbench.models.gies", "GIES", (), {}),
    "gsp": (
        "causalscbench.models.sparsest_permutations",
        "GreedySparsestPermutation",
        (),
        {},
    ),
    "igsp": (
        "causalscbench.models.sparsest_permutations",
        "InterventionalGreedySparsestPermutation",
        (),
        {},
    ),
    "notears-lin-sparse": (
        "causalscbench.models.notears",
        "NotearsLin",
        (),
        {"lambda1": 0.001},
    ),
    "DCDI-G": ("causalscbench.models.dcdi_models", "DCDI", ("DCDI-G",), {}),
    "DCDI-DSF": ("causalscbench.models.dcdi_models", "DCDI", ("DCDI-DSF",), {}),
    "DCDFG-LIN": ("causalscbench.models.dcdi_models", "DCDFG", ("linear",), {}),
    "DCDFG-MLP": ("causalscbench.models.dcdi_models", "DCDFG", ("mlplr",), {}),
    "sortnregress": ("causalscbench.models.varsortability", "Sortnregress", (), {}),
}
TRAINING_INFORMATION = ("observational", "partial_interventional")
MAXIMUM_INPUT_BYTES = 512 * 1024 * 1024
MAXIMUM_ARCHIVE_BYTES = 512 * 1024 * 1024
MAXIMUM_CELLS = 1_000_000
MAXIMUM_GENES = 1_000
CONTROL_LABEL = "non-targeting"
EXPECTED_CAUSALBENCH_COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
EXPECTED_CAUSALBENCH_REPOSITORY = "https://github.com/causalbench/causalbench.git"


class WorkerContractError(ValueError):
    """The allowed input or an external method result breaks the fixed contract."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在隔离环境中运行一种单细胞基因关系方法；输入必须是 HyperSCA "
            "已经允许使用的细胞和基因。"
        )
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        required=True,
        help="含表达矩阵、干预标签和固定基因名的单细胞 NPZ 文件。",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="新建的基因关系结果；已有文件不会被覆盖。",
    )
    parser.add_argument("--model-name", choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--causalbench-source",
        type=Path,
        required=True,
        help="固定提交且没有本地改动的 CausalBench 官方源码目录。",
    )
    parser.add_argument(
        "--training-information",
        choices=TRAINING_INFORMATION,
        required=True,
        help="方法获准读取的是仅未干预细胞，还是公开的部分干预细胞。",
    )
    parser.add_argument("--seed", type=int, required=True, help="固定随机种子。")
    parser.add_argument(
        "--output-semantics",
        choices=("official_return_order", "official_unranked_edges"),
        required=True,
        help=(
            "有官方强弱顺序时赋予递减正分；只有关系集合时全部记为 1.0，"
            "不把容器位置解释为强弱。"
        ),
    )
    return parser


def _regular_snapshot(path: Path) -> bytes:
    source = Path(path).expanduser()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise WorkerContractError(
            "input NPZ must be an existing regular file, not a symbolic link"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkerContractError("input NPZ must be a regular file")
        if metadata.st_size <= 0:
            raise WorkerContractError("input NPZ must not be empty")
        if metadata.st_size > MAXIMUM_INPUT_BYTES:
            raise WorkerContractError("input NPZ is unusually large")
        chunks: list[bytes] = []
        remaining = MAXIMUM_INPUT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size or len(payload) > MAXIMUM_INPUT_BYTES:
            raise WorkerContractError("input NPZ changed while it was being read")
        return payload
    finally:
        os.close(descriptor)


def _canonical_text_vector(values: Any, name: str) -> tuple[str, ...]:
    import numpy as np

    array = np.asarray(values)
    if array.ndim != 1:
        raise WorkerContractError(f"{name} must be one-dimensional")
    if array.dtype.kind not in {"U", "S"}:
        raise WorkerContractError(f"{name} must contain strings without object data")
    if array.dtype.kind == "S":
        try:
            items = tuple(value.decode("utf-8", errors="strict") for value in array.tolist())
        except UnicodeDecodeError as exc:
            raise WorkerContractError(f"{name} must contain valid UTF-8") from exc
    else:
        items = tuple(array.tolist())
    if any(not item for item in items):
        raise WorkerContractError(f"{name} must contain non-empty strings")
    if any(
        any(ord(character) < 32 or ord(character) == 127 for character in item)
        for item in items
    ):
        raise WorkerContractError(f"{name} must not contain an ASCII control character")
    # These values are written to CSV later.  A formula marker is unsafe only
    # at the beginning; an internal hyphen remains a valid part of a gene name.
    if any(item[0] in "=+-@" for item in items):
        raise WorkerContractError(
            f"{name} must not begin with a spreadsheet formula marker"
        )
    if any(item != item.strip() for item in items):
        raise WorkerContractError(f"{name} must not contain surrounding whitespace")
    if any(not unicodedata.is_normalized("NFC", item) for item in items):
        raise WorkerContractError(f"{name} must use NFC text")
    return items


def load_fixed_npz(path: Path) -> tuple[Any, Any, tuple[str, ...]]:
    """Read exactly the three arrays used by the frozen Task C data split."""
    import numpy as np

    payload = _regular_snapshot(path)
    buffer = io.BytesIO(payload)
    try:
        with zipfile.ZipFile(buffer) as archive:
            members = archive.infolist()
            if sum(member.file_size for member in members) > MAXIMUM_ARCHIVE_BYTES:
                raise WorkerContractError("input NPZ has an invalid or unusually large archive")
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as archive:
            base = {"expression_matrix", "interventions", "var_names"}
            allowed = (base, base | {"environment_labels"})
            if set(archive.files) not in allowed:
                raise WorkerContractError(
                    "input NPZ must contain exactly the three base arrays, with only "
                    "the validated environment_labels array optionally added"
                )
            expression = np.asarray(archive["expression_matrix"])
            interventions_raw = np.asarray(archive["interventions"])
            genes_raw = np.asarray(archive["var_names"])
            environment_raw = (
                np.asarray(archive["environment_labels"])
                if "environment_labels" in archive.files
                else None
            )
    except WorkerContractError:
        raise
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise WorkerContractError("input NPZ could not be read safely") from exc

    if expression.ndim != 2:
        raise WorkerContractError("expression_matrix must be two-dimensional")
    rows, columns = expression.shape
    if rows == 0 or rows > MAXIMUM_CELLS:
        raise WorkerContractError("expression_matrix has an invalid number of rows")
    if columns < 2 or columns > MAXIMUM_GENES:
        raise WorkerContractError("expression_matrix has an invalid number of gene columns")
    if expression.dtype.kind not in {"i", "u", "f"}:
        raise WorkerContractError("expression_matrix must contain numeric values")
    if not np.all(np.isfinite(expression)):
        raise WorkerContractError("expression_matrix values must be finite")
    interventions = _canonical_text_vector(interventions_raw, "intervention labels")
    genes = _canonical_text_vector(genes_raw, "gene names")
    if len(interventions) != rows:
        raise WorkerContractError("expression rows must equal intervention labels")
    if len(genes) != columns:
        raise WorkerContractError("expression columns must equal gene names")
    if len(set(genes)) != len(genes):
        raise WorkerContractError("gene names must be unique")
    if CONTROL_LABEL not in interventions:
        raise WorkerContractError("at least one non-targeting cell is required")
    if environment_raw is not None:
        environments = _canonical_text_vector(environment_raw, "environment labels")
        if len(environments) != rows:
            raise WorkerContractError(
                "environment labels must equal expression rows"
            )
        if set(environments) != {"k562", "rpe1"}:
            raise WorkerContractError(
                "environment labels must contain exactly k562 and rpe1"
            )

    safe_expression = np.array(expression, copy=True)
    safe_interventions = np.asarray(interventions, dtype=str)
    safe_expression.setflags(write=False)
    safe_interventions.setflags(write=False)
    return safe_expression, safe_interventions, genes


def select_training_cells(
    expression: Any,
    interventions: Any,
    training_information: str,
) -> tuple[Any, Any]:
    import numpy as np

    if training_information == "observational":
        selected = np.flatnonzero(interventions == CONTROL_LABEL)
        expression = np.array(expression[selected], copy=True)
        interventions = np.array(interventions[selected], copy=True)
    elif training_information == "partial_interventional":
        expression = np.array(expression, copy=True)
        interventions = np.array(interventions, copy=True)
    else:  # The argument parser should make this unreachable.
        raise WorkerContractError("training information is not registered")
    expression.setflags(write=False)
    interventions.setflags(write=False)
    return expression, interventions


def scored_edges(
    raw: object,
    genes: Sequence[str],
    *,
    output_semantics: str,
    limit: int | None = None,
) -> list[tuple[str, str, float]]:
    """Validate official edges and attach only the supported strength evidence."""
    if isinstance(raw, (set, frozenset, Mapping)) or not isinstance(raw, (list, tuple)):
        raise WorkerContractError(
            "failed_invalid_output: official relations must be an explicit list or tuple"
        )
    chosen = raw if limit is None else raw[:limit]
    fixed_genes = set(genes)
    endpoints: list[tuple[str, str]] = []
    for relation in chosen:
        if not isinstance(relation, (list, tuple)) or len(relation) != 2:
            raise WorkerContractError(
                "failed_invalid_output: every returned relation must have two endpoints"
            )
        source, target = relation
        if not isinstance(source, str) or not isinstance(target, str):
            raise WorkerContractError(
                "failed_invalid_output: relation endpoints must be gene names"
            )
        source = str(source)
        target = str(target)
        if (
            not source
            or not target
            or source != source.strip()
            or target != target.strip()
            or not unicodedata.is_normalized("NFC", source)
            or not unicodedata.is_normalized("NFC", target)
        ):
            raise WorkerContractError(
                "failed_invalid_output: relation endpoints must be canonical gene names"
            )
        if source not in fixed_genes or target not in fixed_genes:
            raise WorkerContractError(
                "failed_invalid_output: relation endpoint is outside the fixed gene set"
            )
        endpoints.append((source, target))
    if output_semantics == "official_return_order":
        count = len(endpoints)
        return [
            (source, target, float(count - index))
            for index, (source, target) in enumerate(endpoints)
        ]
    if output_semantics == "official_unranked_edges":
        return [(source, target, 1.0) for source, target in endpoints]
    raise WorkerContractError("failed_invalid_output: output semantics are not supported")


def ranked_edges(
    raw: object, genes: Sequence[str], *, limit: int | None = None
) -> list[tuple[str, str, float]]:
    """Validate a provably ordered edge list and attach decreasing positive scores."""
    return scored_edges(
        raw,
        genes,
        output_semantics="official_return_order",
        limit=limit,
    )


def validate_model_output_semantics(model_name: str, output_semantics: str) -> None:
    expected = MODEL_OUTPUT_SEMANTICS[model_name]
    if output_semantics != expected:
        raise WorkerContractError(
            "failed_invalid_output: requested output semantics does not match "
            f"the registered evidence boundary for {model_name}"
        )


def validate_model_training_information(
    model_name: str, training_information: str
) -> None:
    expected = MODEL_TRAINING_INFORMATION[model_name]
    if training_information != expected:
        raise WorkerContractError(
            "requested training information does not match the registered "
            f"data boundary for {model_name}"
        )


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _git(source: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "CausalBench source cannot be verified as a fixed Git checkout"
        ) from exc
    return completed.stdout.strip()


def _assert_no_symlink_components(path: Path, message: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise SystemExit(message) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(message)


def _reject_git_history_rewrites(source: Path, label: str) -> None:
    replacement_refs = _git(
        source, "for-each-ref", "--format=%(refname)", "refs/replace"
    )
    if replacement_refs:
        raise SystemExit(f"{label} source must not contain Git replace refs")
    grafts_text = _git(source, "rev-parse", "--git-path", "info/grafts")
    grafts = Path(grafts_text)
    if not grafts.is_absolute():
        grafts = source / grafts
    if os.path.lexists(grafts):
        raise SystemExit(f"{label} source must not contain Git grafts")


def validate_causalbench_source(
    source_path: Path, expected_commit: str | None = None
) -> Path:
    """Return the exact verified root of the fixed official source checkout."""
    if expected_commit is None:
        expected_commit = EXPECTED_CAUSALBENCH_COMMIT
    source = Path(os.path.abspath(os.fspath(Path(source_path).expanduser())))
    try:
        metadata = os.lstat(source)
    except OSError as exc:
        raise SystemExit("CausalBench source directory does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise SystemExit("CausalBench source directory must not be a symbolic link")
    if not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit("CausalBench source must be a directory")
    _assert_no_symlink_components(
        source, "CausalBench source path must not use a symbolic link"
    )
    source = source.resolve(strict=True)
    _reject_git_history_rewrites(source, "CausalBench")
    top_level = Path(_git(source, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != source:
        raise SystemExit("CausalBench source must be the verified repository root")
    if _git(source, "rev-parse", "HEAD") != expected_commit:
        raise SystemExit("CausalBench source revision does not match the registered commit")
    if _git(source, "remote", "get-url", "origin") != EXPECTED_CAUSALBENCH_REPOSITORY:
        raise SystemExit("CausalBench source repository does not match the registered source")
    if _git(source, "status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("CausalBench source must have a clean working tree")
    tracked_entries = _git(source, "ls-files", "--stage").splitlines()
    if any(entry.startswith("120000 ") for entry in tracked_entries):
        raise SystemExit("CausalBench source must not contain a symbolic link")
    package = source / "causalscbench"
    try:
        package_metadata = os.lstat(package)
    except OSError as exc:
        raise SystemExit("CausalBench fixed package is missing") from exc
    if stat.S_ISLNK(package_metadata.st_mode) or not stat.S_ISDIR(
        package_metadata.st_mode
    ):
        raise SystemExit(
            "CausalBench fixed package must be a real directory, not a symbolic link"
        )
    return source


def _module_is_within_source(module: object, source: Path) -> bool:
    filename = getattr(module, "__file__", None)
    if not isinstance(filename, str):
        return False
    try:
        module_path = Path(filename).resolve(strict=True)
        module_path.relative_to(source)
        relative = module_path.relative_to(source)
        cursor = source
        for component in relative.parts:
            cursor /= component
            if stat.S_ISLNK(os.lstat(cursor).st_mode):
                return False
    except (OSError, ValueError):
        return False
    return True


def _assert_causalbench_modules_are_verified(source: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name == "causalscbench" or name.startswith("causalscbench."):
            if module is None or not _module_is_within_source(module, source):
                raise WorkerContractError(
                    "CausalBench imported code outside the verified source directory"
                )


@contextmanager
def _verified_causalbench_imports(source: Path) -> Any:
    """Temporarily make the verified checkout the only CausalBench source."""
    previous_path = list(sys.path)
    environment_names = (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
    )
    previous_environment = {name: os.environ.get(name) for name in environment_names}
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "causalscbench" or name.startswith("causalscbench.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)

    user_site = site.getusersitepackages()
    user_sites = {user_site} if isinstance(user_site, str) else set(user_site)
    injected = set(os.environ.get("PYTHONPATH", "").split(os.pathsep))
    clean_path: list[str] = []
    for entry in previous_path:
        candidate = entry or os.getcwd()
        if candidate in user_sites or candidate in injected:
            continue
        clean_path.append(entry)
    sys.path[:] = [str(source), *clean_path]
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        os.environ.pop(name, None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    try:
        yield
        _assert_causalbench_modules_are_verified(source)
    finally:
        for name in tuple(sys.modules):
            if name == "causalscbench" or name.startswith("causalscbench."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_dont_write_bytecode
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _build_selected_model(model_name: str) -> object:
    module_name, class_name, arguments, keywords = MODEL_IMPORTS[model_name]
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class(*arguments, **keywords)


def _absolute_destination(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _open_output_parent(destination: Path) -> tuple[Path, int, tuple[int, int]]:
    """Open/create an absolute parent without following directory symlinks (Linux)."""
    destination = _absolute_destination(destination)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(destination.anchor, flags)
    try:
        for component in destination.parent.parts[1:]:
            try:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        return destination, descriptor, (metadata.st_dev, metadata.st_ino)
    except OSError as exc:
        os.close(descriptor)
        raise WorkerContractError(
            "output parent must be a real directory without symbolic links"
        ) from exc


def _parent_path_matches(parent: Path, identity: tuple[int, int]) -> bool:
    try:
        metadata = os.lstat(parent)
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and (metadata.st_dev, metadata.st_ino) == identity
    )


def _name_exists(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _unlink_same_inode(
    parent_descriptor: int, name: str, identity: tuple[int, int]
) -> bool:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) == identity:
            os.unlink(name, dir_fd=parent_descriptor)
            return True
    except OSError:
        pass
    return False


def write_ranked_csv(path: Path, rows: Sequence[tuple[str, str, float]]) -> None:
    destination, parent_descriptor, parent_identity = _open_output_parent(path)
    target_name = destination.name
    temporary: str | None = None
    temporary_identity: tuple[int, int] | None = None
    linked = False
    completed = False
    try:
        if _name_exists(parent_descriptor, target_name):
            raise WorkerContractError("output CSV must not already exist")
        if not _parent_path_matches(destination.parent, parent_identity):
            raise WorkerContractError("output parent directory changed")
        temporary = f".{target_name}.{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        temporary_identity = (metadata.st_dev, metadata.st_ino)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("source", "target", "score"))
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if not _parent_path_matches(destination.parent, parent_identity):
            raise WorkerContractError("output parent directory changed")
        os.link(
            temporary,
            target_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        linked = True
        if not _parent_path_matches(destination.parent, parent_identity):
            raise WorkerContractError("output parent directory changed")
        if _unlink_same_inode(parent_descriptor, temporary, temporary_identity):
            temporary = None
        else:
            raise WorkerContractError("temporary output could not be removed safely")
        os.fsync(parent_descriptor)
        if not _parent_path_matches(destination.parent, parent_identity):
            raise WorkerContractError("output parent directory changed")
        completed = True
    except FileExistsError as exc:
        raise WorkerContractError("output CSV must not already exist") from exc
    except OSError as exc:
        raise WorkerContractError("output CSV could not be written atomically") from exc
    finally:
        if linked and not completed and temporary_identity is not None:
            _unlink_same_inode(parent_descriptor, target_name, temporary_identity)
        if temporary is not None and temporary_identity is not None:
            _unlink_same_inode(parent_descriptor, temporary, temporary_identity)
        os.close(parent_descriptor)


def validate_output_destination(path: Path) -> None:
    """Reject overwrite attempts before an expensive external method is run."""
    destination, parent_descriptor, parent_identity = _open_output_parent(path)
    try:
        if _name_exists(parent_descriptor, destination.name):
            raise WorkerContractError("output CSV must not already exist")
        if not _parent_path_matches(destination.parent, parent_identity):
            raise WorkerContractError("output parent directory changed")
    finally:
        os.close(parent_descriptor)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        validate_model_output_semantics(args.model_name, args.output_semantics)
        validate_model_training_information(args.model_name, args.training_information)
        validate_output_destination(args.output_csv)
        source = validate_causalbench_source(args.causalbench_source)
        expression, interventions, genes = load_fixed_npz(args.input_npz)
        if args.model_name == "random1000" and len(genes) * (len(genes) - 1) < 1000:
            raise WorkerContractError(
                "random1000 requires at least 1000 possible directed gene relations"
            )
        expression, interventions = select_training_cells(
            expression, interventions, args.training_information
        )
        # Only the selected method module is imported after the checkout and
        # data boundaries are verified; unrelated optional dependencies do not
        # prevent another registered method from running.
        with _verified_causalbench_imports(source):
            training_module = importlib.import_module(
                "causalscbench.models.training_regimes"
            )
            training_regime = training_module.TrainingRegime
            regime = (
                training_regime.Observational
                if args.training_information == "observational"
                else training_regime.PartialIntervational
            )
            model = _build_selected_model(args.model_name)
            _assert_causalbench_modules_are_verified(source)
            returned = model(
                expression,
                interventions.tolist(),
                list(genes),
                regime,
                args.seed,
            )
            _assert_causalbench_modules_are_verified(source)
        rows = scored_edges(
            returned,
            genes,
            output_semantics=args.output_semantics,
        )
        write_ranked_csv(args.output_csv, rows)
    except WorkerContractError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
