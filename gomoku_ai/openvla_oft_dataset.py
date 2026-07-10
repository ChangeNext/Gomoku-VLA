from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DATASET_FORMAT = "gomoku_openvla_oft_multiview_v1"
DEFAULT_INPUT_IMAGE_KEYS = ("board_top_before", "wrist_cam_before")


@dataclass(frozen=True)
class ExportSummary:
    input_jsonl: str
    output_dir: str
    manifest_path: str
    metadata_path: str
    total_records: int
    exported_records: int
    skipped_records: int
    skip_reasons: dict[str, int]


class OpenVLAOFTManifestDataset:
    """Lightweight loader for the custom multi-view OpenVLA-OFT manifest."""

    def __init__(self, manifest_path: str | Path, *, load_images: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.load_images = load_images
        self.records = list(_read_jsonl(self.manifest_path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        if self.load_images:
            images: dict[str, Image.Image] = {}
            for key, relative_path in record["input"]["images"].items():
                images[key] = Image.open(self.root / relative_path).convert("RGB")
            record["input"] = dict(record["input"])
            record["input"]["loaded_images"] = images
        return record


def export_openvla_oft_multiview_dataset(
    input_jsonl: str | Path,
    output_dir: str | Path,
    *,
    input_image_keys: tuple[str, ...] = DEFAULT_INPUT_IMAGE_KEYS,
    copy_images: bool = True,
    include_unusable: bool = False,
    require_so101: bool = True,
) -> ExportSummary:
    """Export raw MuJoCo move records into a custom OpenVLA-OFT multi-view manifest.

    This exporter keeps raw collection artifacts intact and creates a training-facing
    view where only non-leaking model inputs are under ``input``. Strategy and
    execution labels are stored under ``target``.
    """

    source_path = Path(input_jsonl)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.jsonl"
    metadata_path = target_dir / "metadata.json"
    input_root = source_path.parent

    records = list(_read_jsonl(source_path))
    exported = 0
    skip_reasons: dict[str, int] = {}
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for raw_index, raw in enumerate(records):
            reason = _skip_reason(raw, input_image_keys, include_unusable=include_unusable, require_so101=require_so101)
            if reason is not None:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            sample = _build_manifest_sample(
                raw,
                raw_index=raw_index,
                output_dir=target_dir,
                input_root=input_root,
                input_image_keys=input_image_keys,
                copy_images=copy_images,
            )
            manifest_file.write(json.dumps(sample, sort_keys=True) + "\n")
            exported += 1

    metadata = {
        "format": DATASET_FORMAT,
        "input_jsonl": str(source_path),
        "manifest": "manifest.jsonl",
        "input_image_keys": list(input_image_keys),
        "default_model_inputs": {
            "images": list(input_image_keys),
            "language_instruction": "input.language_instruction",
            "state": "input.state",
        },
        "target": {
            "autoregressive_text": "target.text",
            "move_token": "target.move_token",
            "so101_joint_action_sequence": "target.action.sequence",
        },
        "qa_policy": {
            "robot_full_before": "qa only, not a default model input",
            "contact_sheet": "qa only, not a default model input",
        },
        "filters": {
            "include_unusable": include_unusable,
            "require_so101": require_so101,
            "requires_execution_success": not include_unusable,
        },
        "total_records": len(records),
        "exported_records": exported,
        "skipped_records": len(records) - exported,
        "skip_reasons": skip_reasons,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return ExportSummary(
        input_jsonl=str(source_path),
        output_dir=str(target_dir),
        manifest_path=str(manifest_path),
        metadata_path=str(metadata_path),
        total_records=len(records),
        exported_records=exported,
        skipped_records=len(records) - exported,
        skip_reasons=skip_reasons,
    )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _skip_reason(
    raw: dict[str, Any],
    input_image_keys: tuple[str, ...],
    *,
    include_unusable: bool,
    require_so101: bool,
) -> str | None:
    observation = raw.get("observation")
    robot_action = raw.get("robot_action")
    if not isinstance(observation, dict):
        return "missing_observation"
    if not isinstance(robot_action, dict):
        return "missing_robot_action"
    if not include_unusable:
        image_metadata = observation.get("image_metadata", {})
        if not image_metadata.get("training_usable", False):
            return "training_unusable"
        if not raw.get("legal", False):
            return "illegal_move"
        if not robot_action.get("execution_success", False):
            return "execution_failed"
        safety = robot_action.get("safety", {})
        if isinstance(safety, dict) and not safety.get("ok", False):
            return "safety_failed"
        grasp_report = robot_action.get("grasp_report", {})
        if isinstance(grasp_report, dict) and not grasp_report.get("ok", False):
            return "grasp_failed"
    if require_so101 and robot_action.get("controller_type") != "so101_joint_trajectory_v1":
        return "not_so101"
    model_input = observation.get("model_input", {})
    images = model_input.get("images", {}) if isinstance(model_input, dict) else {}
    if not isinstance(images, dict):
        return "missing_model_input_images"
    for key in input_image_keys:
        if key not in images:
            return f"missing_input_image:{key}"
    target_sequence = observation.get("supervision", {}).get("target_sequence", {})
    if not target_sequence.get("tokens"):
        return "missing_target_sequence"
    action = observation.get("supervision", {}).get("execution", {}).get("action", {})
    if not action.get("sequence"):
        return "missing_action_sequence"
    return None


def _build_manifest_sample(
    raw: dict[str, Any],
    *,
    raw_index: int,
    output_dir: Path,
    input_root: Path,
    input_image_keys: tuple[str, ...],
    copy_images: bool,
) -> dict[str, Any]:
    observation = raw["observation"]
    robot_action = raw["robot_action"]
    model_input = observation["model_input"]
    supervision = observation["supervision"]
    target_sequence = supervision["target_sequence"]
    execution = supervision["execution"]
    strategy = supervision["strategy"]
    sample_id = _sample_id(raw, raw_index)
    sample_dir = output_dir / "samples" / sample_id
    images = _export_images(
        model_input["images"],
        keys=input_image_keys,
        input_root=input_root,
        output_dir=output_dir,
        sample_dir=sample_dir / "inputs",
        copy_images=copy_images,
    )
    qa_images = _export_qa_images(
        observation,
        input_root=input_root,
        output_dir=output_dir,
        sample_dir=sample_dir / "qa",
        copy_images=copy_images,
    )
    tokens = [str(token) for token in target_sequence["tokens"]]
    return {
        "format": DATASET_FORMAT,
        "sample_id": sample_id,
        "source": {
            "jsonl_index": raw_index,
            "game_id": raw["game_id"],
            "step": raw["step"],
            "timestamp": raw.get("timestamp"),
            "policy_source": raw.get("policy_source"),
            "checkpoint": raw.get("checkpoint"),
        },
        "input": {
            "language_instruction": model_input["language_instruction"],
            "image_order": list(input_image_keys),
            "images": images,
            "state": model_input.get("state", {}),
        },
        "target": {
            "format": target_sequence["format"],
            "text": " ".join(tokens),
            "tokens": tokens,
            "move_token": target_sequence["move_token"],
            "move": strategy["selected_move"],
            "action_index": raw["action_index"],
            "policy_probs": raw.get("policy_probs", []),
            "value": raw.get("value"),
            "action": {
                "type": "so101_joint_trajectory",
                "tokenization": target_sequence["action_tokenization"],
                "names": execution["action"]["format"],
                "sequence": execution["action"]["sequence"],
                "controller_type": execution["action"]["controller_type"],
            },
        },
        "quality": {
            "legal": raw.get("legal", False),
            "training_usable": observation.get("image_metadata", {}).get("training_usable", False),
            "execution_success": robot_action.get("execution_success", False),
            "safety_ok": robot_action.get("safety", {}).get("ok") if isinstance(robot_action.get("safety"), dict) else None,
            "grasp_report": robot_action.get("grasp_report"),
            "placement_error_cell": robot_action.get("placement_error_cell"),
            "placement_error_world": robot_action.get("placement_error_world"),
        },
        "qa": {
            "images": qa_images,
            "all_raw_images": observation.get("images", {}),
        },
    }


def _export_images(
    images: dict[str, str],
    *,
    keys: tuple[str, ...],
    input_root: Path,
    output_dir: Path,
    sample_dir: Path,
    copy_images: bool,
) -> dict[str, str]:
    exported: dict[str, str] = {}
    for key in keys:
        exported[key] = _export_image(
            images[key],
            filename=f"{key}.png",
            input_root=input_root,
            output_dir=output_dir,
            sample_dir=sample_dir,
            copy_images=copy_images,
        )
    return exported


def _export_qa_images(
    observation: dict[str, Any],
    *,
    input_root: Path,
    output_dir: Path,
    sample_dir: Path,
    copy_images: bool,
) -> dict[str, str]:
    qa_images: dict[str, str] = {}
    all_images = observation.get("images", {})
    if isinstance(all_images, dict) and "robot_full_before" in all_images:
        qa_images["robot_full_before"] = _export_image(
            all_images["robot_full_before"],
            filename="robot_full_before.png",
            input_root=input_root,
            output_dir=output_dir,
            sample_dir=sample_dir,
            copy_images=copy_images,
        )
    contact_sheet = observation.get("image_metadata", {}).get("qa_contact_sheet")
    if isinstance(contact_sheet, str) and contact_sheet:
        qa_images["contact_sheet"] = _export_image(
            contact_sheet,
            filename="contact_sheet.png",
            input_root=input_root,
            output_dir=output_dir,
            sample_dir=sample_dir,
            copy_images=copy_images,
        )
    return qa_images


def _export_image(
    source: str,
    *,
    filename: str,
    input_root: Path,
    output_dir: Path,
    sample_dir: Path,
    copy_images: bool,
) -> str:
    source_path = _resolve_source_path(source, input_root)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not copy_images:
        return _relative_or_string(source_path, output_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    target_path = sample_dir / filename
    shutil.copy2(source_path, target_path)
    return target_path.relative_to(output_dir).as_posix()


def _resolve_source_path(path_value: str, input_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return input_root / path


def _relative_or_string(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _sample_id(raw: dict[str, Any], raw_index: int) -> str:
    game_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(raw.get("game_id", "game")))
    step = int(raw.get("step", raw_index))
    return f"{game_id}_step_{step:04d}"
