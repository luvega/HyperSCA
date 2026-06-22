from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

from scripts.prepare_osta_benchmark_cache import build_dataset


def _write_tenx_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("data", data=np.asarray([1, 2, 3], dtype=np.int32))
        matrix.create_dataset("indices", data=np.asarray([0, 2, 1], dtype=np.int64))
        matrix.create_dataset("indptr", data=np.asarray([0, 2, 3], dtype=np.int64))
        matrix.create_dataset("shape", data=np.asarray([3, 2], dtype=np.int32))
        matrix.create_dataset("barcodes", data=np.asarray([b"cellid_000000003-1", b"cellid_000000004-1"]))
        features = matrix.create_group("features")
        features.create_dataset("name", data=np.asarray([b"GENE1", b"GENE2", b"GENE3"]))
        features.create_dataset("id", data=np.asarray([b"ENSG1", b"ENSG2", b"ENSG3"]))
        features.create_dataset("feature_type", data=np.asarray([b"Gene Expression"] * 3))
        features.create_dataset("genome", data=np.asarray([b"GRCh38"] * 3))


def test_build_dataset_writes_visiumhd_segmented_cache(tmp_path: Path):
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    segmented = raw_root / "VisiumHD_HumanColon_Oliveira" / "segmented_outputs"
    _write_tenx_h5(segmented / "filtered_feature_cell_matrix.h5")
    (segmented / "cell_segmentations.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"cell_id": 3},
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"cell_id": 4},
                        "geometry": {"type": "Polygon", "coordinates": [[[10, 0], [12, 0], [12, 2], [10, 2], [10, 0]]]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_dataset(
        raw_root,
        cache_root,
        "VisiumHD_HumanColon_Oliveira_segmented",
        overwrite=True,
    )

    assert manifest["status"] == "written"
    assert manifest["assay_type"] == "visiumhd_segmented_cell"
    cached = ad.read_h5ad(cache_root / "VisiumHD_HumanColon_Oliveira_segmented" / "benchmark.h5ad")
    assert cached.shape == (2, 3)
    assert cached.obs["cell_id_numeric"].tolist() == [3, 4]
    assert cached.obs["cell_area_px2"].tolist() == [4.0, 4.0]
    assert cached.obs[["x", "y"]].round(3).values.tolist() == [[1.0, 1.0], [11.0, 1.0]]
    assert "spatial" in cached.obsm
