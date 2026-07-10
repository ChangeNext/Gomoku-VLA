from __future__ import annotations

import argparse
from pathlib import Path

from gomoku_ai.openvla_oft_dataset import (
    DEFAULT_INPUT_IMAGE_KEYS,
    export_openvla_oft_multiview_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MuJoCo Gomoku records to a custom OpenVLA-OFT multi-view manifest."
    )
    parser.add_argument("--input-jsonl", required=True, help="Raw MuJoCo policy episodes JSONL.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--input-images",
        default=",".join(DEFAULT_INPUT_IMAGE_KEYS),
        help="Comma-separated model input image keys. Defaults to board_top_before,wrist_cam_before.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Keep source image paths instead of copying images into per-sample folders.",
    )
    parser.add_argument(
        "--include-unusable",
        action="store_true",
        help="Include records that failed training usability/execution filters.",
    )
    parser.add_argument(
        "--allow-non-so101",
        action="store_true",
        help="Allow non-SO-101 controller records. Not recommended for the OpenVLA-OFT SO-101 path.",
    )
    args = parser.parse_args()

    image_keys = tuple(key.strip() for key in args.input_images.split(",") if key.strip())
    if not image_keys:
        parser.error("--input-images must contain at least one image key")

    summary = export_openvla_oft_multiview_dataset(
        Path(args.input_jsonl),
        Path(args.output_dir),
        input_image_keys=image_keys,
        copy_images=not args.no_copy_images,
        include_unusable=args.include_unusable,
        require_so101=not args.allow_non_so101,
    )
    print(
        "exported "
        f"{summary.exported_records}/{summary.total_records} records to {summary.manifest_path}; "
        f"skipped={summary.skipped_records} reasons={summary.skip_reasons}",
        flush=True,
    )


if __name__ == "__main__":
    main()
