from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.validate_so101_dataset import REQUIRED_INPUT_IMAGES, _find_forbidden_keys


def validate_manifest(path: Path, *, expected_samples: int | None = None) -> dict[str, Any]:
    records = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")

    seen_ids: set[str] = set()
    missing_images = 0
    corrupt_images = 0
    leakage_records = 0
    for index, record in enumerate(records):
        prefix = f"sample[{index}]"
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in seen_ids:
            errors.append(f"{prefix}: missing or duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)

        model_input = record.get("input") or {}
        images = model_input.get("images") or {}
        if set(images) != REQUIRED_INPUT_IMAGES:
            errors.append(f"{prefix}: input image keys are {sorted(images)}")
        leaked = sorted(_find_forbidden_keys(model_input))
        if leaked:
            leakage_records += 1
            errors.append(f"{prefix}: target leakage in input keys {leaked}")
        instruction = str(model_input.get("language_instruction", ""))
        if "<MOVE_" in instruction or "target" in instruction.lower():
            leakage_records += 1
            errors.append(f"{prefix}: target-like instruction {instruction!r}")

        target = record.get("target") or {}
        tokens = target.get("tokens") or []
        action = target.get("action") or {}
        if not tokens or tokens[0] != target.get("move_token") or tokens[-1] != "<EOS>":
            errors.append(f"{prefix}: invalid autoregressive target token sequence")
        if action.get("controller_type") != "so101_joint_trajectory_v1":
            errors.append(f"{prefix}: target is not an SO-101 joint trajectory")
        names = action.get("names") or []
        sequence = action.get("sequence") or []
        if not names or not sequence or not all(
            isinstance(point, list)
            and len(point) == len(names)
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
            for point in sequence
        ):
            errors.append(f"{prefix}: invalid continuous action sequence")

        quality = record.get("quality") or {}
        if not all(quality.get(key) is True for key in ("legal", "training_usable", "execution_success", "safety_ok")):
            errors.append(f"{prefix}: failed record was included in training manifest")
        if (quality.get("grasp_report") or {}).get("ok") is not True:
            errors.append(f"{prefix}: failed grasp was included in training manifest")

        for group in (images, (record.get("qa") or {}).get("images") or {}):
            for value in group.values():
                image_path = path.parent / str(value)
                if not image_path.exists():
                    missing_images += 1
                    errors.append(f"{prefix}: missing exported image {value}")
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, ValueError) as exc:
                    corrupt_images += 1
                    errors.append(f"{prefix}: corrupt exported image {value}: {exc}")

    if expected_samples is not None and len(records) != expected_samples:
        errors.append(f"expected {expected_samples} samples, found {len(records)}")
    return {
        "status": "pass" if not errors else "fail",
        "samples": len(records),
        "missing_images": missing_images,
        "corrupt_images": corrupt_images,
        "input_leakage_records": leakage_records,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an exported Gomoku OpenVLA manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    summary = validate_manifest(Path(args.manifest), expected_samples=args.expected_samples)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"validated {summary['samples']} exported samples: {summary['status']}", flush=True)
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
