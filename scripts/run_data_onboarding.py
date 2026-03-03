#!/usr/bin/env python
"""Phase D0: 四项目数据标准化入库总入口。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validators import validate_onboarding_tree


def _run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _run_optional(cmd: list[str], label: str) -> bool:
    """Run a command; on failure print warning but return False instead of raising."""
    print(f"[CMD] {label}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        print(f"[WARN] {label} failed (rc={proc.returncode}), continuing ...")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HyperSCA Phase D0 onboarding")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--icb-root", default=r"G:\scCRC_ICB")
    parser.add_argument("--neu-root", default=r"G:\scCRC_Neu")
    parser.add_argument("--st-root", default=r"G:\ST_CRC_MSS")
    parser.add_argument("--ifng-root", default=r"F:\scCRC_IFNG")
    parser.add_argument("--skip-schema", action="store_true")
    parser.add_argument(
        "--build-reference", action="store_true",
        help="After data ingest, train ICB reference model (scVI/scANVI) → data/ref",
    )
    parser.add_argument("--ref-label-key", default=None,
                        help="Cell-type label column for reference (default: auto-detect)")
    parser.add_argument("--ref-max-epochs", type=int, default=100)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    # 1) 入库并标准化为 /data/scRNA + /data/ST
    _run(
        [
            sys.executable,
            "scripts/prepare_h5ad.py",
            "--mode",
            "research",
            "--icb-root",
            args.icb_root,
            "--neu-root",
            args.neu_root,
            "--st-root",
            args.st_root,
            "--ifng-root",
            args.ifng_root,
        ]
    )

    # 2) 统一 schema 到 /data/metadata
    if not args.skip_schema:
        _run(
            [
                sys.executable,
                "scripts/build_canonical_schema.py",
                "--data-root",
                str(data_root),
                "--neu-root",
                args.neu_root,
                "--ifng-root",
                args.ifng_root,
                "--icb-root",
                str(Path(args.icb_root) / "output"),
                "--st-root",
                args.st_root,
            ]
        )

    # 3) Build ICB reference model (optional, non-blocking on failure)
    if args.build_reference:
        ref_cmd = [
            sys.executable, "scripts/build_icb_reference.py",
            "--max-epochs", str(args.ref_max_epochs),
        ]
        if args.ref_label_key:
            ref_cmd.extend(["--label-key", args.ref_label_key])
        _run_optional(ref_cmd, "build_icb_reference")

    # 4) 目录完整性校验
    issues = validate_onboarding_tree(data_root)
    if issues:
        print("[WARN] Onboarding validation issues:")
        for item in issues:
            print("  -", item)
        return 1

    print("[DONE] Phase D0 onboarding completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
