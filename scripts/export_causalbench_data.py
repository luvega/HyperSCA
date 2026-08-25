"""使用固定版本 CausalBench 生成 K562 和 RPE1 官方缓存。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPOSITORY = "https://github.com/causalbench/causalbench.git"
COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
DATASETS = ["dataset_k562.npz", "dataset_rpe1.npz"]
REFERENCES = [
    "reference_k562_pooled.csv",
    "reference_k562_chipseq.csv",
    "reference_rpe1_pooled.csv",
    "reference_rpe1_chipseq.csv",
]
KNOWN_SOURCE_FILES = (
    "k562.h5ad", "rpe1.h5ad", "summary_stats.xlsx", "corum_complexes.txt.zip",
    "human_lr_pair.txt", "protein.links.txt.gz", "protein.physical.links.txt.gz",
    "protein.info.txt.gz",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载并整理任务 C 的 K562/RPE1 官方单细胞干预数据。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="保存官方数据缓存、参考关系和导出清单的目录。",
    )
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        help=(
            "正式导出读取已核对原始 H5AD 和辅助缓存的目录；"
            "必须与 --data-dir 的全新版本目录分开。"
        ),
    )
    parser.add_argument(
        "--method-assets-root",
        type=Path,
        help=(
            "正式导出使用 bootstrap_task_c_methods.py 固定并核对的官方源码目录。"
        ),
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="请求 CausalBench 对表达数据进行官方过滤。",
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="仅输出固定来源说明，不下载数据或导入 CausalBench。",
    )
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        help="独立保存的公开镜像与 H5AD 转换核对记录。",
    )
    parser.add_argument(
        "--require-acquisition-manifest",
        action="store_true",
        help="正式数据导出必须提供并核对 --acquisition-manifest。",
    )
    return parser


def _description(data_dir: Path, use_filter: bool) -> dict[str, object]:
    datasets = [
        name.replace(".npz", "_filtered.npz") if use_filter else name
        for name in DATASETS
    ]
    return {
        "schema_version": "1.0",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "datasets": datasets,
        "references": REFERENCES,
        "data_dir": str(data_dir),
        "filter": bool(use_filter),
    }


def _write_atomic(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_edges(path: Path, edges: set[tuple[str, str]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["source", "target"])
    writer.writerows(sorted(edges))
    _write_atomic(path, output.getvalue())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _relative_path(path: Path, data_dir: Path) -> str:
    return os.path.relpath(path.resolve(), data_dir.resolve())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    description = _description(args.data_dir, args.filter)
    if args.describe_only:
        print(json.dumps(description, ensure_ascii=False, sort_keys=True))
        return 0

    if args.require_acquisition_manifest and args.acquisition_manifest is None:
        raise SystemExit(
            "正式数据导出缺少独立的获取记录：请提供 --acquisition-manifest。"
        )
    formal_mode = args.require_acquisition_manifest or args.acquisition_manifest is not None
    if formal_mode:
        if args.source_data_dir is None:
            raise SystemExit(
                "正式数据导出缺少原始输入目录：请提供 --source-data-dir。"
            )
        if args.method_assets_root is None:
            raise SystemExit(
                "正式数据导出缺少固定官方源码：请提供 --method-assets-root。"
            )

        from src.evaluation.task_c_formal_export import (
            TaskCFormalExportError,
            export_task_c_formal_bundle,
        )

        assert args.acquisition_manifest is not None
        try:
            summary = export_task_c_formal_bundle(
                source_data_dir=args.source_data_dir,
                output_dir=args.data_dir,
                acquisition_manifest=args.acquisition_manifest,
                method_assets_root=args.method_assets_root,
                use_filter=bool(args.filter),
            )
        except TaskCFormalExportError as exc:
            raise SystemExit(f"正式数据导出无效：{exc}") from exc
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    try:
        from causalscbench.data_access.create_dataset import CreateDataset
        from causalscbench.data_access.create_evaluation_datasets import (
            CreateEvaluationDatasets,
        )
    except ImportError as exc:
        raise SystemExit(
            "无法导入固定版本 CausalBench；请先创建 envs/task_c/causalbench.yml 中的环境。"
        ) from exc

    args.data_dir.mkdir(parents=True, exist_ok=True)
    k562_path, rpe1_path = CreateDataset(
        str(args.data_dir), bool(args.filter)
    ).load()
    dataset_paths = {
        "k562": str(Path(k562_path).resolve()),
        "rpe1": str(Path(rpe1_path).resolve()),
    }
    description["paths"] = dataset_paths
    reference_paths: dict[str, str] = {}
    for context_id, dataset_name in (
        ("k562", "weissmann_k562"),
        ("rpe1", "weissmann_rpe1"),
    ):
        corum, ligand_receptor, string_network, string_physical, chipseq = (
            CreateEvaluationDatasets(str(args.data_dir), dataset_name).load()
        )
        pooled = set().union(
            corum, ligand_receptor, string_network, string_physical, chipseq
        )
        pooled_self = {edge for edge in pooled if edge[0] == edge[1]}
        pooled = pooled - pooled_self
        pooled_bidir = pooled | {(target, source) for source, target in pooled}
        chipseq_self = {edge for edge in chipseq if edge[0] == edge[1]}
        for reference_id, edges in (
            ("pooled", pooled_bidir),
            ("chipseq", set(chipseq) - chipseq_self),
        ):
            path = args.data_dir / f"reference_{context_id}_{reference_id}.csv"
            _write_edges(path, edges)
            reference_paths[f"{context_id}_{reference_id}"] = str(path.resolve())
        description.setdefault("dropped_self_edges", {})[context_id] = {
            "pooled": len(pooled_self), "chipseq": len(chipseq_self)
        }

    description["reference_paths"] = reference_paths
    description["reference_scope"] = {
        "pooled": "CausalBench pooled biological evidence expanded in both directions",
        "chipseq": (
            "CausalBench bundled directed ChIP evidence; the RPE1 branch uses "
            "the bundled HepG2 file in this pinned commit"
        ),
    }
    description["reference_sources"] = {
        "corum": "https://mips.helmholtz-muenchen.de/corum/download/releases/current/humanComplexes.txt.zip",
        "ligand_receptor": (
            "https://raw.githubusercontent.com/LewisLabUCSD/Ligand-Receptor-Pairs/"
            "ba44c3c4b4a3e501667309dd9ce7208501aeb961/Human/Human-2020-Shao-LR-pairs.txt"
        ),
        "string_network": "https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz",
        "string_physical": "https://stringdb-static.org/download/protein.physical.links.detailed.v11.5/9606.protein.physical.links.detailed.v11.5.txt.gz",
        "chip_atlas_license": "https://dbarchive.biosciencedbc.jp/en/chip-atlas/lic.html",
        "k562_dataset": "https://plus.figshare.com/ndownloader/files/35773219",
        "rpe1_dataset": "https://plus.figshare.com/ndownloader/files/35775606",
        "string_protein_info": "https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz",
    }
    description["downloaded_at_utc"] = None
    description["acquisition_time_note"] = (
        "CausalBench may reuse existing caches; acquisition time is not provable from this export."
    )
    description["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    artifact_paths = [Path(path) for path in dataset_paths.values()] + [
        Path(path) for path in reference_paths.values()
    ]
    description["sha256"] = {
        _relative_path(path, args.data_dir): _sha256(path) for path in artifact_paths
    }
    description["source_sha256"] = {
        _relative_path(args.data_dir / name, args.data_dir): _sha256(args.data_dir / name)
        for name in KNOWN_SOURCE_FILES if (args.data_dir / name).is_file()
    }
    manifest_path = args.data_dir / "export_manifest.json"
    _write_atomic(
        manifest_path,
        json.dumps(description, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(description, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
