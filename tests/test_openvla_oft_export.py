import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from gomoku_ai.openvla_oft_dataset import (
    DATASET_FORMAT,
    OpenVLAOFTManifestDataset,
    export_openvla_oft_multiview_dataset,
)


class OpenVLAOFTExportTest(unittest.TestCase):
    def test_export_multiview_manifest_keeps_robot_full_as_qa_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            assets = root / "assets"
            assets.mkdir()
            for name in ("board_top_before", "wrist_cam_before", "robot_full_before", "contact_sheet"):
                Image.new("RGB", (16, 16), (20, 40, 60)).save(assets / f"{name}.png")

            input_jsonl = root / "episodes.jsonl"
            input_jsonl.write_text(json.dumps(_record(root)) + "\n", encoding="utf-8")
            output_dir = root / "openvla_oft"

            summary = export_openvla_oft_multiview_dataset(input_jsonl, output_dir)

            self.assertEqual(summary.total_records, 1)
            self.assertEqual(summary.exported_records, 1)
            self.assertEqual(summary.skipped_records, 0)
            manifest_row = json.loads((output_dir / "manifest.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(manifest_row["format"], DATASET_FORMAT)
            self.assertEqual(manifest_row["input"]["image_order"], ["board_top_before", "wrist_cam_before"])
            self.assertEqual(set(manifest_row["input"]["images"]), {"board_top_before", "wrist_cam_before"})
            self.assertNotIn("robot_full_before", manifest_row["input"]["images"])
            self.assertIn("robot_full_before", manifest_row["qa"]["images"])
            self.assertIn("contact_sheet", manifest_row["qa"]["images"])
            self.assertTrue((output_dir / manifest_row["input"]["images"]["board_top_before"]).exists())
            self.assertTrue((output_dir / manifest_row["input"]["images"]["wrist_cam_before"]).exists())
            self.assertEqual(manifest_row["target"]["text"], "<MOVE_004> <ACT_SO101_0000> <EOS>")
            self.assertEqual(manifest_row["target"]["action"]["names"], ["shoulder_pan", "gripper"])
            self.assertEqual(manifest_row["target"]["action"]["sequence"], [[0.1, 1.0]])

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["format"], DATASET_FORMAT)
            self.assertEqual(metadata["input_image_keys"], ["board_top_before", "wrist_cam_before"])

            dataset = OpenVLAOFTManifestDataset(output_dir / "manifest.jsonl", load_images=True)
            self.assertEqual(len(dataset), 1)
            loaded = dataset[0]
            self.assertEqual(set(loaded["input"]["loaded_images"]), {"board_top_before", "wrist_cam_before"})
            self.assertEqual(loaded["input"]["loaded_images"]["board_top_before"].size, (16, 16))

    def test_export_skips_unusable_and_non_so101_records_by_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            assets = root / "assets"
            assets.mkdir()
            for name in ("board_top_before", "wrist_cam_before"):
                Image.new("RGB", (16, 16)).save(assets / f"{name}.png")

            records = [_record(root), _record(root)]
            records[0]["observation"]["image_metadata"]["training_usable"] = False
            records[1]["robot_action"]["controller_type"] = "scripted_kinematic_v1"
            input_jsonl = root / "episodes.jsonl"
            input_jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            summary = export_openvla_oft_multiview_dataset(input_jsonl, root / "openvla_oft")

            self.assertEqual(summary.exported_records, 0)
            self.assertEqual(summary.skipped_records, 2)
            self.assertEqual(summary.skip_reasons["training_unusable"], 1)
            self.assertEqual(summary.skip_reasons["not_so101"], 1)


def _record(root: Path) -> dict:
    assets = root / "assets"
    return {
        "game_id": "game-1",
        "step": 0,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "board_before": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "board_after": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "board_size": 3,
        "win_length": 3,
        "rule_set": "free",
        "enforce_center_opening": False,
        "current_player": "black",
        "current_player_value": 1,
        "selected_move": [1, 1],
        "action_index": 4,
        "policy_source": "test",
        "policy_probs": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "value": 0.25,
        "legal": True,
        "used_tactical_move": False,
        "winner": None,
        "winner_value": None,
        "terminal": False,
        "checkpoint": "checkpoint.pt",
        "robot_action": {
            "controller_type": "so101_joint_trajectory_v1",
            "execution_success": True,
            "placement_error_cell": 0.0,
            "placement_error_world": 0.0,
            "safety": {"ok": True},
            "grasp_report": {"ok": True},
        },
        "observation": {
            "image_metadata": {
                "training_usable": True,
                "qa_contact_sheet": str(assets / "contact_sheet.png"),
            },
            "model_input": {
                "language_instruction": "play the strongest legal Gomoku move as black",
                "images": {
                    "board_top_before": str(assets / "board_top_before.png"),
                    "wrist_cam_before": str(assets / "wrist_cam_before.png"),
                },
                "state": {
                    "board_flat": [0] * 9,
                    "current_player_value": 1,
                    "robot_model": "so101",
                },
            },
            "images": {
                "board_top_before": str(assets / "board_top_before.png"),
                "wrist_cam_before": str(assets / "wrist_cam_before.png"),
                "robot_full_before": str(assets / "robot_full_before.png"),
            },
            "supervision": {
                "target_sequence": {
                    "format": "autoregressive_move_then_action_v1",
                    "tokens": ["<MOVE_004>", "<ACT_SO101_0000>", "<EOS>"],
                    "move_token": "<MOVE_004>",
                    "action_tokenization": "so101_joint_tokens_v1",
                },
                "strategy": {
                    "selected_move": [1, 1],
                },
                "execution": {
                    "action": {
                        "format": ["shoulder_pan", "gripper"],
                        "sequence": [[0.1, 1.0]],
                        "controller_type": "so101_joint_trajectory_v1",
                    },
                },
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
