"""Freeze the audited no-release outcome for methods protocol v2.1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.evaluation.methods_protocol_outcome import (
        outcome_from_pilot_summary,
        write_protocol_outcome_exclusively,
    )

    write_protocol_outcome_exclusively(
        args.output,
        outcome_from_pilot_summary(args.pilot_summary),
    )


if __name__ == "__main__":
    main()
