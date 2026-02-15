"""一键运行全部 HyperSCA Example"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
EXAMPLES = [
    ("Example 01: Chromium Metadata QC", "run_example_01.py"),
    ("Example 02: Visium Spatial Graph", "run_example_02.py"),
    ("Example 03: VisiumHD Segmentation", "run_example_03.py"),
    ("Example 04: Xenium Panel Summary", "run_example_04.py"),
]


def main():
    print("=" * 60)
    print("HyperSCA Examples - Batch Runner")
    print("=" * 60)

    # 日志文件
    log_dir = SCRIPTS_DIR.parent / "results" / "examples"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_log.txt"

    results = []
    total_start = time.time()

    for title, script in EXAMPLES:
        print(f"\n{'-' * 40}")
        print(f">> {title}")
        print(f"{'-' * 40}")
        t0 = time.time()
        ret = subprocess.run(
            [PYTHON, str(SCRIPTS_DIR / script)],
            cwd=str(SCRIPTS_DIR.parent),
        )
        elapsed = time.time() - t0
        status = "OK" if ret.returncode == 0 else f"FAIL (code={ret.returncode})"
        results.append((title, status, f"{elapsed:.1f}s"))
        print(f"  [{status}] {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    # 写日志
    lines = ["HyperSCA Examples Run Log", "=" * 40, ""]
    for title, status, elapsed in results:
        lines.append(f"{title}: {status} ({elapsed})")
    lines.append("")
    lines.append(f"Total: {total_elapsed:.1f}s")
    n_ok = sum(1 for _, s, _ in results if s == "OK")
    lines.append(f"Passed: {n_ok}/{len(results)}")
    log_path.write_text("\n".join(lines), encoding="utf-8")

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"完成: {n_ok}/{len(results)} 通过, 总耗时 {total_elapsed:.1f}s")
    print(f"日志: {log_path}")
    print(f"{'=' * 60}")

    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
