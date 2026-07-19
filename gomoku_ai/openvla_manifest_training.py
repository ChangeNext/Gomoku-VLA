from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .openvla_finetuning_prep import OpenVLAFinetuningManifestAdapter


IGNORE_INDEX = -100
DEFAULT_PROMPT_TEMPLATE = "In: {instruction}\nOut:"


@dataclass(frozen=True)
class OpenVLATrainingBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    labels: torch.Tensor


class OpenVLAMoveOnlyDataset(Dataset[dict[str, Any]]):
    """Map the custom Gomoku OpenVLA manifest to HF OpenVLA training samples."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        processor: Any,
        stage: str = "move_only",
        board_size: int = 15,
        image_key: str = "board_top_before",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_length: int = 256,
    ) -> None:
        self.adapter = OpenVLAFinetuningManifestAdapter(manifest_path, stage=stage, board_size=board_size)
        self.manifest_root = Path(manifest_path).parent
        self.processor = processor
        self.image_key = image_key
        self.prompt_template = prompt_template
        self.max_length = max_length
        if max_length <= 0:
            raise ValueError("max_length must be positive")

    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.adapter[index]
        if self.image_key not in sample.image_paths:
            raise ValueError(f"sample {sample.sample_id!r} does not contain image {self.image_key!r}")
        prompt = self.prompt_template.format(instruction=sample.instruction)
        text = f"{prompt} {sample.target_text}"
        image_path = self.manifest_root / sample.image_paths[self.image_key]
        image = Image.open(image_path).convert("RGB")
        encoded = self.processor(
            text=text,
            images=image,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_ids = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"][0]
        input_ids = encoded["input_ids"][0]
        labels = input_ids.clone()
        labels[: min(len(prompt_ids), len(labels))] = IGNORE_INDEX
        labels[encoded["attention_mask"][0] == 0] = IGNORE_INDEX
        return {
            "sample_id": sample.sample_id,
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"][0],
            "pixel_values": encoded["pixel_values"][0],
            "labels": labels,
            "target_text": sample.target_text,
            "prompt": prompt,
        }


def collate_openvla_training_batch(
    samples: list[dict[str, Any]],
    *,
    pad_token_id: int,
    padding_side: str = "right",
) -> OpenVLATrainingBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    max_length = max(int(sample["input_ids"].numel()) for sample in samples)
    input_ids = []
    attention_mask = []
    labels = []
    for sample in samples:
        ids = sample["input_ids"]
        mask = sample["attention_mask"]
        label = sample["labels"]
        pad_length = max_length - int(ids.numel())
        if padding_side == "right":
            input_ids.append(torch.nn.functional.pad(ids, (0, pad_length), value=pad_token_id))
            attention_mask.append(torch.nn.functional.pad(mask, (0, pad_length), value=0))
            labels.append(torch.nn.functional.pad(label, (0, pad_length), value=IGNORE_INDEX))
        else:
            input_ids.append(torch.nn.functional.pad(ids, (pad_length, 0), value=pad_token_id))
            attention_mask.append(torch.nn.functional.pad(mask, (pad_length, 0), value=0))
            labels.append(torch.nn.functional.pad(label, (pad_length, 0), value=IGNORE_INDEX))
    return OpenVLATrainingBatch(
        input_ids=torch.stack(input_ids),
        attention_mask=torch.stack(attention_mask),
        pixel_values=torch.stack([sample["pixel_values"] for sample in samples]),
        labels=torch.stack(labels),
    )


def write_training_summary(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
