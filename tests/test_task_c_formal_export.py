from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.evaluation import task_c_acquisition as acquisition_module
from src.evaluation import task_c_formal_export as formal_export_module
from src.evaluation.task_c_acquisition import AcquisitionFileSpec
from src.evaluation.task_c_formal_export import (
    TaskCFormalExportError,
    _validate_npz,
    export_task_c_formal_bundle,
)


SUPPORT_FILES = (
    "corum_complexes.txt.zip",
    "human_lr_pair.txt",
    "protein.links.txt.gz",
    "protein.physical.links.txt.gz",
    "protein.info.txt.gz",
)

FAKE_CAUSALBENCH_EVIDENCE = {
    "repository": "https://github.com/causalbench/causalbench.git",
    "commit": "1a2143cffdc85f835b41ce8d52034be1bf903e71",
    "bootstrap_manifest_sha256": "sha256:" + "1" * 64,
    "source_worktree_sha256": "sha256:" + "2" * 64,
    "private_source_inventory_sha256": "sha256:" + "3" * 64,
}


def _snapshot_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _install_fake_causalbench_asset_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> types.ModuleType:
    hook = types.ModuleType("hypersca_test_causalbench_hook")
    hook.events = []
    hook.tamper_module = False
    hook.fail_load = False
    monkeypatch.setitem(sys.modules, hook.__name__, hook)
    inventories: dict[str, str] = {}

    def validate(*args, **kwargs):
        del args, kwargs
        return dict(FAKE_CAUSALBENCH_EVIDENCE)

    def materialize(*args, **kwargs):
        destination = Path(args[1])
        expected = kwargs["expected_evidence"]
        assert dict(expected) == FAKE_CAUSALBENCH_EVIDENCE
        package = destination / "causalscbench/data_access"
        package.mkdir(parents=True)
        (destination / "causalscbench/__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "create_dataset.py").write_text(
            """from pathlib import Path
import anndata as ad
import numpy as np
import hypersca_test_causalbench_hook as hook
hook.events.append('load_causalbench')
class CreateDataset:
    def __init__(self, data_dir, use_filter):
        assert use_filter is False
        self.data_dir = Path(data_dir)
    def load(self):
        hook.events.append('create_datasets')
        if hook.fail_load:
            raise RuntimeError('simulated CausalBench failure')
        if hook.tamper_module:
            with open(__file__, 'a', encoding='utf-8') as handle:
                handle.write('\\n# changed during execution\\n')
        paths = []
        for context in ('k562', 'rpe1'):
            source = self.data_dir / f'{context}.h5ad'
            assert source.is_symlink()
            assert __import__('os').readlink(source).startswith('/proc/self/fd/')
            dataset = ad.read_h5ad(source, backed='r')
            try:
                assert dataset.shape == (3, 2)
            finally:
                dataset.file.close()
            path = self.data_dir / f'dataset_{context}.npz'
            np.savez(
                path,
                expression_matrix=np.asarray(
                    [[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]], dtype=np.float32
                ),
                interventions=np.asarray(
                    ['non-targeting', 'ENSG000001', 'ENSG000002']
                ),
                var_names=np.asarray(['ENSG000001', 'ENSG000002']),
            )
            paths.append(str(path))
        return tuple(paths)
""",
            encoding="utf-8",
        )
        (package / "create_evaluation_datasets.py").write_text(
            """from pathlib import Path
import hypersca_test_causalbench_hook as hook
class CreateEvaluationDatasets:
    def __init__(self, data_dir, dataset_name):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
    def load(self):
        hook.events.append(f'references:{self.dataset_name}')
        return ({('ENSG000001', 'ENSG000002')}, set(), set(), set(), {('ENSG000001', 'ENSG000002')})
""",
            encoding="utf-8",
        )
        inventories[str(destination.resolve())] = _snapshot_tree_digest(destination)
        return dict(FAKE_CAUSALBENCH_EVIDENCE)

    def verify(source_root, evidence):
        assert dict(evidence) == FAKE_CAUSALBENCH_EVIDENCE
        root = Path(source_root)
        if _snapshot_tree_digest(root) != inventories[str(root.resolve())]:
            raise formal_export_module.TaskCRuntimeError(
                "private CausalBench source snapshot changed"
            )

    def remove(source_root, evidence):
        verify(source_root, evidence)
        shutil.rmtree(source_root)

    monkeypatch.setattr(
        formal_export_module, "validate_causalbench_export_assets", validate
    )
    monkeypatch.setattr(
        formal_export_module, "materialize_causalbench_source_snapshot", materialize
    )
    monkeypatch.setattr(
        formal_export_module, "verify_causalbench_source_snapshot", verify
    )
    monkeypatch.setattr(
        formal_export_module, "remove_causalbench_source_snapshot", remove
    )
    return hook


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


def _write_acquisition_inputs(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    _install_fake_causalbench_asset_boundary(monkeypatch)
    source = root / "raw"
    source.mkdir()
    obs = pd.DataFrame(
        {"gene_id": ["ENSG000001", "ENSG000002", "ENSG000001"]},
        index=["cell-1", "cell-2", "cell-3"],
    )
    original_var = pd.DataFrame(
        {"ensembl_id": ["ENSG000001", "ENSG000002"], "chr": ["1", "2"]},
        index=["GENE1", "GENE2"],
    )
    expression = np.asarray([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    mirrors: dict[str, Path] = {}
    converted: dict[str, Path] = {}
    specs: dict[str, AcquisitionFileSpec] = {}
    for context in ("k562", "rpe1"):
        mirror = root / f"{context}-official.h5ad"
        converted_path = source / f"{context}.h5ad"
        ad.AnnData(X=expression, obs=obs, var=original_var).write_h5ad(mirror)
        converted_var = original_var.copy()
        converted_var["gene_name"] = converted_var.index.to_numpy(copy=True)
        converted_var.index = converted_var["ensembl_id"].to_numpy(copy=True)
        ad.AnnData(X=expression, obs=obs, var=converted_var).write_h5ad(
            converted_path
        )
        mirrors[context] = mirror
        converted[context] = converted_path
        specs[context] = AcquisitionFileSpec(
            context_id=context,
            file_name=mirror.name,
            size_bytes=mirror.stat().st_size,
            md5=_md5(mirror),
            zenodo_content_url=(
                f"https://zenodo.org/api/records/7041849/files/{mirror.name}/content"
            ),
            figshare_original_url=f"https://plus.figshare.com/{context}",
        )
    monkeypatch.setattr(acquisition_module, "OFFICIAL_ACQUISITION_FILES", specs)
    for name in SUPPORT_FILES:
        (source / name).write_bytes(f"fixed source cache: {name}\n".encode("utf-8"))
    # These stale outputs must never be copied into the versioned formal export.
    (source / "dataset_k562.npz").write_bytes(b"stale-old-dataset")
    (source / "reference_k562_pooled.csv").write_bytes(b"stale-old-reference")
    acquisition = root / "acquisition_manifest.json"
    acquisition_module.create_task_c_acquisition_manifest(
        mirror_paths=mirrors,
        converted_paths=converted,
        output_path=acquisition,
        authoritative_files=specs,
        requested_chunk_rows=2,
    )
    return source, acquisition


def _run_formal_export(
    root: Path,
    source: Path,
    acquisition: Path,
    events: list[str],
) -> dict[str, object]:
    hook = sys.modules["hypersca_test_causalbench_hook"]
    hook.events = events
    return export_task_c_formal_bundle(
        source_data_dir=source,
        output_dir=root / "raw_export_v2",
        acquisition_manifest=acquisition,
        method_assets_root=root / "method-assets",
        use_filter=False,
    )


def test_formal_export_stages_exact_files_and_ignores_old_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, acquisition = _write_acquisition_inputs(tmp_path, monkeypatch)
    events: list[str] = []

    summary = _run_formal_export(tmp_path, source, acquisition, events)

    output = tmp_path / "raw_export_v2"
    assert summary["reuse_status"] == "created_new_formal_export"
    assert set(path.name for path in output.iterdir()) == {
        "dataset_k562.npz",
        "dataset_rpe1.npz",
        "reference_k562_pooled.csv",
        "reference_k562_chipseq.csv",
        "reference_rpe1_pooled.csv",
        "reference_rpe1_chipseq.csv",
        "export_manifest.json",
    }
    assert (output / "dataset_k562.npz").read_bytes() != b"stale-old-dataset"
    assert (output / "reference_k562_pooled.csv").read_bytes() != (
        b"stale-old-reference"
    )
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["schema_version"] == "2.0"
    assert manifest["status"] == "formal_export_complete"
    assert set(manifest["artifact_sha256"]) == set(path.name for path in output.iterdir()) - {
        "export_manifest.json"
    }
    assert set(manifest["supporting_source_files"]) == set(SUPPORT_FILES)
    assert manifest["causalbench_source"] == FAKE_CAUSALBENCH_EVIDENCE
    assert all(not Path(value).is_absolute() for value in manifest["paths"].values())
    assert not list(tmp_path.glob(".raw_export_v2.staging-*"))
    assert events == [
        "load_causalbench",
        "create_datasets",
        "references:weissmann_k562",
        "references:weissmann_rpe1",
    ]


def test_existing_formal_export_is_reused_only_after_complete_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, acquisition = _write_acquisition_inputs(tmp_path, monkeypatch)
    first_events: list[str] = []
    _run_formal_export(tmp_path, source, acquisition, first_events)
    second_events: list[str] = []

    summary = _run_formal_export(tmp_path, source, acquisition, second_events)

    assert summary["reuse_status"] == "verified_existing_formal_export"
    assert second_events == []


def test_partial_stale_symlink_and_hardlinked_formal_outputs_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, acquisition = _write_acquisition_inputs(tmp_path, monkeypatch)
    partial = tmp_path / "raw_export_v2"
    partial.mkdir()
    (partial / "dataset_k562.npz").write_bytes(b"partial")
    before = (partial / "dataset_k562.npz").read_bytes()
    with pytest.raises(TaskCFormalExportError, match="partial|complete|existing"):
        _run_formal_export(tmp_path, source, acquisition, [])
    assert (partial / "dataset_k562.npz").read_bytes() == before
    assert not list(tmp_path.glob(".raw_export_v2.staging-*"))

    other_root = tmp_path / "hardlink-case"
    other_root.mkdir()
    source2, acquisition2 = _write_acquisition_inputs(other_root, monkeypatch)
    _run_formal_export(other_root, source2, acquisition2, [])
    artifact = other_root / "raw_export_v2/dataset_k562.npz"
    os.link(artifact, other_root / "outside-hardlink.npz")
    with pytest.raises(TaskCFormalExportError, match="link|regular"):
        _run_formal_export(other_root, source2, acquisition2, [])

    link_root = tmp_path / "symlink-case"
    link_root.mkdir()
    source3, acquisition3 = _write_acquisition_inputs(link_root, monkeypatch)
    elsewhere = link_root / "elsewhere"
    elsewhere.mkdir()
    (link_root / "raw_export_v2").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(TaskCFormalExportError, match="symbolic|directory"):
        _run_formal_export(link_root, source3, acquisition3, [])


def test_failed_generation_and_parent_fsync_leave_no_formal_or_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, acquisition = _write_acquisition_inputs(tmp_path, monkeypatch)

    hook = sys.modules["hypersca_test_causalbench_hook"]
    hook.fail_load = True

    with pytest.raises(RuntimeError, match="simulated"):
        export_task_c_formal_bundle(
            source_data_dir=source,
            output_dir=tmp_path / "raw_export_v2",
            acquisition_manifest=acquisition,
            method_assets_root=tmp_path / "method-assets",
            use_filter=False,
        )
    hook.fail_load = False
    assert not (tmp_path / "raw_export_v2").exists()
    assert not list(tmp_path.glob(".raw_export_v2.staging-*"))

    from src.evaluation import task_c_formal_export as export_module

    monkeypatch.setattr(
        export_module,
        "_fsync_parent_after_publish",
        lambda descriptor: (_ for _ in ()).throw(OSError("simulated fsync failure")),
    )
    with pytest.raises(TaskCFormalExportError, match="fsync|publish"):
        _run_formal_export(tmp_path, source, acquisition, [])
    assert not (tmp_path / "raw_export_v2").exists()
    assert not list(tmp_path.glob(".raw_export_v2.staging-*"))


def test_formal_npz_rejects_nonfinite_expression_without_loading_the_matrix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset_k562.npz"
    np.savez(
        path,
        expression_matrix=np.asarray([[0.0, np.nan]], dtype=np.float32),
        interventions=np.asarray(["non-targeting"]),
        var_names=np.asarray(["ENSG000001", "ENSG000002"]),
    )

    with pytest.raises(TaskCFormalExportError, match="non-finite"):
        _validate_npz(path)


def test_existing_formal_export_rejects_duplicate_manifest_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, acquisition = _write_acquisition_inputs(tmp_path, monkeypatch)
    _run_formal_export(tmp_path, source, acquisition, [])
    manifest = tmp_path / "raw_export_v2/export_manifest.json"
    payload = manifest.read_text(encoding="utf-8")
    manifest.chmod(0o600)
    manifest.write_text(
        payload.replace("{", '{"schema_version":"2.0",', 1),
        encoding="utf-8",
    )

    with pytest.raises(TaskCFormalExportError, match="manifest|schema"):
        _run_formal_export(tmp_path, source, acquisition, [])


@pytest.mark.parametrize("change_point", ["before_rename", "after_rename"])
def test_acquisition_change_around_publish_never_leaves_a_formal_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_point: str,
) -> None:
    source, acquisition = _write_acquisition_inputs(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement-acquisition.json"
    replacement.write_bytes(acquisition.read_bytes() + b" ")
    changed = False

    def replace_acquisition() -> None:
        nonlocal changed
        if not changed:
            os.replace(replacement, acquisition)
            changed = True

    real_fsync = formal_export_module.os.fsync
    real_rename = formal_export_module._rename_noreplace

    def fsync_then_change(descriptor: int) -> None:
        real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if change_point == "before_rename" and os.path.basename(
            os.readlink(f"/proc/self/fd/{descriptor}")
        ).startswith(".raw_export_v2.staging-") and os.path.isdir(
            f"/proc/self/fd/{descriptor}"
        ):
            assert metadata.st_mode
            replace_acquisition()

    def rename_then_change(parent: int, source_name: str, target_name: str) -> None:
        real_rename(parent, source_name, target_name)
        if change_point == "after_rename":
            replace_acquisition()

    monkeypatch.setattr(formal_export_module.os, "fsync", fsync_then_change)
    monkeypatch.setattr(formal_export_module, "_rename_noreplace", rename_then_change)

    with pytest.raises(TaskCFormalExportError, match="acquisition|变化|替换"):
        _run_formal_export(tmp_path, source, acquisition, [])

    assert changed is True
    assert not (tmp_path / "raw_export_v2").exists()
    assert not list(tmp_path.glob(".raw_export_v2.staging-*"))


def test_private_causalbench_module_tamper_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, acquisition = _write_acquisition_inputs(tmp_path, monkeypatch)
    hook = sys.modules["hypersca_test_causalbench_hook"]
    hook.tamper_module = True

    with pytest.raises(TaskCFormalExportError, match="source|源码|快照|changed"):
        _run_formal_export(tmp_path, source, acquisition, [])

    assert not (tmp_path / "raw_export_v2").exists()
    assert not list(tmp_path.glob(".raw_export_v2.staging-*"))
