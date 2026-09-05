#!/usr/bin/env python3
"""Materialize one seed's exact benchmark splits for inspection/reuse."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rare_event.data import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, n, offset, split, a, b in [
        ("train", 40000, 1000, "train", 0, 0), ("calibration", 20000, 2000, "calibration", 0, 0),
        ("policy", 20000, 3000, "policy", 0, 0), ("id", 40000, 4000, "test", 0, 0),
        ("combined", 40000, 4000, "test", 1, 1)]:
        frame = generate(n, .005, args.seed+offset, split, a, b)
        frame.to_json(args.output / f"{name}.jsonl.gz", orient="records", lines=True, compression="gzip")
    print(args.output.resolve())
