"""使用固定版本 CausalBench 生成 K562 和 RPE1 官方缓存。"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "https://github.com/causalbench/causalbench.git"
COMMIT = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
DATASETS = ["dataset_k562.npz", "dataset_rpe1.npz"]
REFERENCES = [
    "reference_k562_pooled.csv",
    "reference_k562_chipseq.csv",
    "reference_rpe1_pooled.csv",
    "reference_rpe1_chipseq.csv",
]


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
        "--filter",
        action="store_true",
        help="请求 CausalBench 对表达数据进行官方过滤。",
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="仅输出固定来源说明，不下载数据或导入 CausalBench。",
    )
    return parser


def _description(data_dir: Path, use_filter: bool) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "datasets": DATASETS,
        "references": REFERENCES,
        "data_dir": str(data_dir),
        "filter": bool(use_filter),
    }


def _write_edges(path: Path, edges: set[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target"])
        writer.writerows(sorted(edges))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    description = _description(args.data_dir, args.filter)
    if args.describe_only:
        print(json.dumps(description, ensure_ascii=False, sort_keys=True))
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
    description["paths"] = {
        "k562": str(Path(k562_path).resolve()),
        "rpe1": str(Path(rpe1_path).resolve()),
    }

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
        pooled_bidir = pooled | {(target, source) for source, target in pooled}
        for reference_id, edges in (
            ("pooled", pooled_bidir),
            ("chipseq", set(chipseq)),
        ):
            path = args.data_dir / f"reference_{context_id}_{reference_id}.csv"
            _write_edges(path, edges)
            reference_paths[f"{context_id}_{reference_id}"] = str(path.resolve())

    description["reference_paths"] = reference_paths
    description["reference_scope"] = {
        "pooled": "CausalBench pooled biological evidence expanded in both directions",
        "chipseq": (
            "CausalBench bundled directed ChIP evidence; the RPE1 branch uses "
            "the bundled HepG2 file in this pinned commit"
        ),
    }
    description["reference_sources"] = {
        "corum": "https://mips.helmholtz-muenchen.de/corum/",
        "ligand_receptor": (
            "https://github.com/LewisLabUCSD/Ligand-Receptor-Pairs/"
            "tree/ba44c3c4b4a3e501667309dd9ce7208501aeb961"
        ),
        "string_db": "https://string-db.org/cgi/download.pl",
        "chip_atlas": "https://dbarchive.biosciencedbc.jp/en/chip-atlas/lic.html",
    }
    description["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.data_dir / "export_manifest.json").write_text(
        json.dumps(description, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(description, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
