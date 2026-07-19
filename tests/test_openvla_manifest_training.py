import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from gomoku_ai.openvla_manifest_training import (
    IGNORE_INDEX,
    OpenVLAMoveOnlyDataset,
    collate_openvla_training_batch,
)


class OpenVLAManifestTrainingTest(unittest.TestCase):
    def test_dataset_masks_prompt_labels_and_collates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = _write_manifest(root)
            dataset = OpenVLAMoveOnlyDataset(manifest, processor=_FakeProcessor(), board_size=15)

            sample = dataset[0]
            self.assertEqual(sample["target_text"], "<MOVE_112> <EOS>")
            self.assertTrue(torch.any(sample["labels"] != IGNORE_INDEX))
            self.assertEqual(sample["labels"][0].item(), IGNORE_INDEX)
            self.assertEqual(tuple(sample["pixel_values"].shape), (6, 8, 8))

            batch = collate_openvla_training_batch([sample], pad_token_id=0)
            self.assertEqual(tuple(batch.input_ids.shape), (1, sample["input_ids"].numel()))
            self.assertEqual(tuple(batch.pixel_values.shape), (1, 6, 8, 8))


class _FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text, return_tensors, padding=False, truncation=True, max_length=256):
        tokens = [1] + [abs(hash(part)) % 100 + 2 for part in str(text).split()]
        tokens = tokens[:max_length]
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
        }


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def __call__(self, text, images, padding=False, truncation=True, max_length=256, return_tensors="pt"):
        encoded = self.tokenizer(text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length)
        encoded["pixel_values"] = torch.zeros((1, 6, 8, 8), dtype=torch.float32)
        return encoded


def _write_manifest(root: Path) -> Path:
    image_dir = root / "samples" / "sample-1" / "inputs"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_dir / "board_top_before.png")
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_dir / "wrist_cam_before.png")
    record = {
        "sample_id": "sample-1",
        "input": {
            "language_instruction": "play the strongest legal Gomoku move as black",
            "image_order": ["board_top_before", "wrist_cam_before"],
            "images": {
                "board_top_before": "samples/sample-1/inputs/board_top_before.png",
                "wrist_cam_before": "samples/sample-1/inputs/wrist_cam_before.png",
            },
            "state": {"board_flat": [0] * 225, "current_player_value": 1},
        },
        "target": {
            "tokens": ["<MOVE_112>", "<ACT_SO101_0000>", "<EOS>"],
            "move_token": "<MOVE_112>",
            "action_index": 112,
            "move": [7, 7],
            "text": "<MOVE_112> <ACT_SO101_0000> <EOS>",
        },
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    unittest.main()
