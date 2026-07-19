from __future__ import annotations

import argparse
import json
from pathlib import Path

from gomoku_ai.openvla_manifest_split import split_openvla_manifest_by_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Split an OpenVLA manifest into train/val/test by game id.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", help="Defaults to the manifest directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-prefix", default="manifest")
    args = parser.parse_args()

    summary = split_openvla_manifest_by_game(
        Path(args.manifest),
        Path(args.output_dir) if args.output_dir else None,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
