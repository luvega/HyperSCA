"""核验 K562/RPE1，并生成任务 C 的五个固定数据划分。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.task_c_data import (  # noqa: E402
    TaskCDataError,
    build_shared_task_c_split,
    build_task_c_provenance,
    build_task_c_reference_provenance,
    check_task_c_json_record,
    check_task_c_materializations,
    load_task_c_dataset,
    materialize_task_c_splits,
    write_task_c_json_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="核验任务 C 官方数据，并隔离模型建立与最终检验所用细胞。"
    )
    parser.add_argument("--k562-npz", type=Path, required=True)
    parser.add_argument("--rpe1-npz", type=Path, required=True)
    parser.add_argument("--k562-pooled-reference", type=Path, required=True)
    parser.add_argument("--k562-chipseq-reference", type=Path, required=True)
    parser.add_argument("--rpe1-pooled-reference", type=Path, required=True)
    parser.add_argument("--rpe1-chipseq-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-cells-per-intervention", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        k562 = load_task_c_dataset(args.k562_npz, context_id="k562")
        rpe1 = load_task_c_dataset(args.rpe1_npz, context_id="rpe1")
        provenance_dir = args.output_dir / "provenance"
        provenance_records = [
            (provenance_dir / "k562.json", build_task_c_provenance(k562)),
            (provenance_dir / "rpe1.json", build_task_c_provenance(rpe1)),
        ]
        for context_id in ("k562", "rpe1"):
            provenance_records.append(
                (
                    provenance_dir / f"{context_id}_references.json",
                    build_task_c_reference_provenance(
                        context_id=context_id,
                        pooled_path=getattr(args, f"{context_id}_pooled_reference"),
                        chipseq_path=getattr(args, f"{context_id}_chipseq_reference"),
                    ),
                )
            )
        requests = []
        for seed in (11, 23, 47, 71, 97):
            split = build_shared_task_c_split(
                k562,
                rpe1,
                seed=seed,
                min_cells=args.min_cells_per_intervention,
            )
            requests.append((split, args.output_dir / "splits" / f"seed_{seed}"))

        for path, payload in provenance_records:
            check_task_c_json_record(path, payload)
        check_task_c_materializations(k562, rpe1, requests)
        for path, payload in provenance_records:
            write_task_c_json_record(path, payload)
        results = materialize_task_c_splits(k562, rpe1, requests)

        summaries = []
        for (split, _), result in zip(requests, results):
            summaries.append(
                {
                    "seed": split.seed,
                    "split_id": split.split_id,
                    "public_manifest": result["public_manifest"],
                }
            )
    except TaskCDataError as exc:
        parser.error(f"无法准备任务 C 数据：{exc}")
    print(json.dumps({"status": "prepared", "splits": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
