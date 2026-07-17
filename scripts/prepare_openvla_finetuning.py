from __future__ import annotations

import argparse
import json
from pathlib import Path

from gomoku_ai.openvla_finetuning_prep import (
    DEFAULT_STAGE,
    VALID_STAGES,
    prepare_openvla_finetuning_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare GPU-side OpenVLA fine-tuning integration files without running training."
    )
    parser.add_argument("--manifest", required=True, help="Exported OpenVLA-OFT manifest.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated preparation files.")
    parser.add_argument("--base-model", required=True, help="OpenVLA/OFT base model identifier or local path.")
    parser.add_argument(
        "--stage",
        choices=sorted(VALID_STAGES),
        default=DEFAULT_STAGE,
        help="Training stage to describe in the generated config.",
    )
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--preview-samples", type=int, default=5)
    args = parser.parse_args()

    summary = prepare_openvla_finetuning_package(
        Path(args.manifest),
        Path(args.output_dir),
        base_model=args.base_model,
        stage=args.stage,
        board_size=args.board_size,
        preview_samples=args.preview_samples,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
