from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.evaluation.task_c_acquisition import (
    AcquisitionFileSpec,
    OFFICIAL_ACQUISITION_FILES,
    TaskCAcquisitionError,
    create_task_c_acquisition_manifest,
    load_task_c_acquisition_manifest,
    verify_export_sources_against_acquisition,
    verify_h5ad_conversion,
)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - source record


def _write_h5ad_pair(
    root: Path,
    context: str,
    *,
    sparse_matrix: bool = False,
) -> tuple[Path, Path]:
    expression = np.asarray(
        [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 4.0, 5.0]],
        dtype=np.float32,
    )
    if sparse_matrix:
        expression = sparse.csr_matrix(expression)  # type: ignore[assignment]
    obs = pd.DataFrame(
        {
            "perturbation": pd.Categorical(["control", "A", "B"]),
            "batch": ["one", "one", "two"],
        },
        index=["cell-1", "cell-2", "cell-3"],
    )
    var = pd.DataFrame(
        {
            "ensembl_id": ["ENSG000001", "ENSG000002", "ENSG000003"],
            "chromosome": ["1", "1", "2"],
            "measured": [True, True, True],
        },
        index=["GENE1", "GENE2", "GENE3"],
    )
    mirror = root / f"{context}-mirror.h5ad"
    converted = root / f"{context}-converted.h5ad"
    ad.AnnData(X=expression, obs=obs, var=var).write_h5ad(mirror)
    converted_var = var.copy()
    converted_var["gene_name"] = converted_var.index.to_numpy(copy=True)
    converted_var.index = converted_var["ensembl_id"].to_numpy(copy=True)
    ad.AnnData(X=expression, obs=obs, var=converted_var).write_h5ad(converted)
    return mirror, converted


def _spec(context: str, mirror: Path) -> AcquisitionFileSpec:
    return AcquisitionFileSpec(
        context_id=context,
        file_name=mirror.name,
        size_bytes=mirror.stat().st_size,
        md5=_md5(mirror),
        zenodo_content_url=(
            f"https://zenodo.org/api/records/7041849/files/{mirror.name}/content"
        ),
        figshare_original_url=f"https://plus.figshare.com/{context}",
    )


@pytest.mark.parametrize("sparse_matrix", [False, True])
def test_h5ad_conversion_is_checked_in_bounded_row_chunks(
    tmp_path: Path,
    sparse_matrix: bool,
) -> None:
    mirror, converted = _write_h5ad_pair(
        tmp_path, "k562", sparse_matrix=sparse_matrix
    )

    result = verify_h5ad_conversion(mirror, converted, requested_chunk_rows=2)

    assert result["shape"] == [3, 3]
    assert result["expression_equal"] is True
    assert result["obs_equal"] is True
    assert result["var_conversion_equal"] is True
    assert result["chunk_rows"] == 2
    assert result["matrix_storage"] == ("csr" if sparse_matrix else "dense")


def test_h5ad_conversion_rejects_changed_values_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    mirror, converted = _write_h5ad_pair(tmp_path, "rpe1")
    changed = ad.read_h5ad(converted)
    changed.X[1, 2] = np.nan
    changed.write_h5ad(converted)

    with pytest.raises(TaskCAcquisitionError, match="finite|expression"):
        verify_h5ad_conversion(mirror, converted, requested_chunk_rows=2)


def test_h5ad_validation_reopens_only_descriptor_bound_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror, converted = _write_h5ad_pair(tmp_path, "k562")
    evil_root = tmp_path / "replacement"
    evil_root.mkdir()
    evil_mirror, evil_converted = _write_h5ad_pair(evil_root, "k562")
    evil = ad.read_h5ad(evil_mirror)
    evil.X[0, 0] = 91.0
    evil.write_h5ad(evil_mirror)
    evil = ad.read_h5ad(evil_converted)
    evil.X[0, 0] = 91.0
    evil.write_h5ad(evil_converted)
    real_read_h5ad = ad.read_h5ad
    originals = (mirror, converted)
    replacements = (evil_mirror, evil_converted)
    backups = tuple(tmp_path / f"original-{index}.h5ad" for index in range(2))
    swapped = False
    opened = 0

    def swapping_read_h5ad(path, *args, **kwargs):
        nonlocal swapped, opened
        candidate = Path(path)
        if candidate in originals and not swapped:
            for original, replacement, backup in zip(
                originals, replacements, backups, strict=True
            ):
                os.replace(original, backup)
                os.replace(replacement, original)
            swapped = True
        result = real_read_h5ad(path, *args, **kwargs)
        opened += 1
        if swapped and opened == 2:
            for original, replacement, backup in zip(
                originals, replacements, backups, strict=True
            ):
                os.replace(original, replacement)
                os.replace(backup, original)
        return result

    monkeypatch.setattr(ad, "read_h5ad", swapping_read_h5ad)

    verify_h5ad_conversion(mirror, converted, requested_chunk_rows=2)

    assert swapped is False


def test_acquisition_manifest_binds_fixed_metadata_and_local_files(
    tmp_path: Path,
) -> None:
    mirrors: dict[str, Path] = {}
    converted: dict[str, Path] = {}
    specs: dict[str, AcquisitionFileSpec] = {}
    for context in ("k562", "rpe1"):
        mirror, transformed = _write_h5ad_pair(tmp_path, context)
        mirrors[context] = mirror
        converted[context] = transformed
        specs[context] = _spec(context, mirror)
    evidence = tmp_path / "k562-403.html"
    evidence.write_text("<h1>403 Forbidden</h1>", encoding="utf-8")
    output = tmp_path / "acquisition_manifest.json"

    summary = create_task_c_acquisition_manifest(
        mirror_paths=mirrors,
        converted_paths=converted,
        output_path=output,
        figshare_403_evidence={"k562": evidence},
        authoritative_files=specs,
        requested_chunk_rows=2,
    )

    assert summary["status"] == "verified_local_acquisition_and_conversion"
    payload, reference = load_task_c_acquisition_manifest(output)
    assert reference["sha256"].startswith("sha256:")
    assert payload["zenodo"] == {
        "doi": "10.5281/zenodo.7041849",
        "license": "CC-BY-4.0",
        "record_id": 7041849,
        "record_url": "https://zenodo.org/records/7041849",
    }
    assert payload["local_time_semantics"] == (
        "filesystem_observed_mtime_ns is local file metadata, not server download time"
    )
    assert payload["datasets"]["k562"]["mirror"]["md5"] == specs["k562"].md5
    assert payload["datasets"]["k562"]["figshare_403_evidence"][
        "sha256"
    ].startswith("sha256:")
    assert payload["datasets"]["rpe1"]["figshare_403_evidence"] is None
    assert "downloaded_at" not in json.dumps(payload)

    with pytest.raises(TaskCAcquisitionError, match="overwrite|exists"):
        create_task_c_acquisition_manifest(
            mirror_paths=mirrors,
            converted_paths=converted,
            output_path=output,
            authoritative_files=specs,
            requested_chunk_rows=2,
        )


def test_acquisition_rejects_symlinks_hardlinks_and_wrong_official_digest(
    tmp_path: Path,
) -> None:
    mirror, converted = _write_h5ad_pair(tmp_path, "k562")
    linked = tmp_path / "linked.h5ad"
    linked.symlink_to(mirror)
    with pytest.raises(TaskCAcquisitionError, match="symbolic|regular"):
        verify_h5ad_conversion(linked, converted)

    hardlinked = tmp_path / "hardlinked.h5ad"
    os.link(mirror, hardlinked)
    with pytest.raises(TaskCAcquisitionError, match="链接|link"):
        verify_h5ad_conversion(mirror, converted)
    hardlinked.unlink()

    wrong_k562 = _spec("k562", mirror)
    wrong_k562 = AcquisitionFileSpec(
        context_id=wrong_k562.context_id,
        file_name=wrong_k562.file_name,
        size_bytes=wrong_k562.size_bytes,
        md5="0" * 32,
        zenodo_content_url=wrong_k562.zenodo_content_url,
        figshare_original_url=wrong_k562.figshare_original_url,
    )
    wrong_rpe1 = AcquisitionFileSpec(
        context_id="rpe1",
        file_name=mirror.name,
        size_bytes=mirror.stat().st_size,
        md5="0" * 32,
        zenodo_content_url=(
            f"https://zenodo.org/api/records/7041849/files/{mirror.name}/content"
        ),
        figshare_original_url="https://plus.figshare.com/rpe1",
    )
    with pytest.raises(TaskCAcquisitionError, match="MD5|official"):
        create_task_c_acquisition_manifest(
            mirror_paths={"k562": mirror, "rpe1": mirror},
            converted_paths={"k562": converted, "rpe1": converted},
            output_path=tmp_path / "wrong.json",
            authoritative_files={"k562": wrong_k562, "rpe1": wrong_rpe1},
        )


def test_export_source_check_rejects_files_changed_after_acquisition_record(
    tmp_path: Path,
) -> None:
    mirrors: dict[str, Path] = {}
    converted: dict[str, Path] = {}
    specs: dict[str, AcquisitionFileSpec] = {}
    for context in ("k562", "rpe1"):
        mirror, transformed = _write_h5ad_pair(tmp_path, context)
        mirrors[context] = mirror
        converted[context] = transformed
        specs[context] = _spec(context, mirror)
    manifest = tmp_path / "acquisition.json"
    create_task_c_acquisition_manifest(
        mirror_paths=mirrors,
        converted_paths=converted,
        output_path=manifest,
        authoritative_files=specs,
    )
    payload, _ = load_task_c_acquisition_manifest(manifest)

    verify_export_sources_against_acquisition(payload, converted)
    changed = ad.read_h5ad(converted["k562"])
    changed.X[0, 0] = 19.0
    changed.write_h5ad(converted["k562"])
    with pytest.raises(TaskCAcquisitionError, match="acquisition record"):
        verify_export_sources_against_acquisition(payload, converted)


def test_acquisition_reader_rejects_incomplete_conversion_evidence(
    tmp_path: Path,
) -> None:
    mirrors: dict[str, Path] = {}
    converted: dict[str, Path] = {}
    specs: dict[str, AcquisitionFileSpec] = {}
    for context in ("k562", "rpe1"):
        mirror, transformed = _write_h5ad_pair(tmp_path, context)
        mirrors[context] = mirror
        converted[context] = transformed
        specs[context] = _spec(context, mirror)
    manifest = tmp_path / "acquisition.json"
    create_task_c_acquisition_manifest(
        mirror_paths=mirrors,
        converted_paths=converted,
        output_path=manifest,
        authoritative_files=specs,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["datasets"]["k562"]["conversion_rule"]
    manifest.chmod(0o600)
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(TaskCAcquisitionError, match="schema|conversion|转换"):
        load_task_c_acquisition_manifest(manifest)


def test_official_zenodo_file_metadata_is_frozen() -> None:
    assert OFFICIAL_ACQUISITION_FILES["k562"].size_bytes == 1_546_729_675
    assert OFFICIAL_ACQUISITION_FILES["k562"].md5 == (
        "d8cba17576d1a8afc0f7d71b79cad0f7"
    )
    assert OFFICIAL_ACQUISITION_FILES["rpe1"].size_bytes == 1_236_886_900
    assert OFFICIAL_ACQUISITION_FILES["rpe1"].md5 == (
        "cc7f1ec50aeb3a3e1b4a6cfa713d80fa"
    )
