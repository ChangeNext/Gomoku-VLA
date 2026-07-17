import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gomoku_ai.openvla_finetuning_prep import (
    OpenVLAFinetuningManifestAdapter,
    action_index_to_move_token,
    apply_special_tokens_to_tokenizer,
    build_move_tokens,
    legal_move_token_mask,
    mask_move_logits,
    prepare_openvla_finetuning_package,
)


class OpenVLAFinetuningPrepTest(unittest.TestCase):
    def test_prepare_package_writes_tokens_config_prompt_and_preview(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(_manifest_record()) + "\n", encoding="utf-8")

            output_dir = root / "prep"
            summary = prepare_openvla_finetuning_package(
                manifest,
                output_dir,
                base_model="openvla/openvla-7b",
                stage="move_only",
            )

            self.assertEqual(summary.samples, 1)
            self.assertEqual(summary.move_tokens, 225)
            special_tokens = json.loads((output_dir / "special_tokens.json").read_text(encoding="utf-8"))
            self.assertEqual(special_tokens["move_tokens"][0], "<MOVE_000>")
            self.assertEqual(special_tokens["move_tokens"][-1], "<MOVE_224>")
            self.assertIn("<ACT_SO101_0000>", special_tokens["action_tokens"])

            run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertTrue(run_config["tokenizer"]["resize_embeddings"])
            self.assertTrue(run_config["tokenizer"]["train_new_embedding_rows"])
            self.assertEqual(run_config["loss"]["recommended_start_lambdas"]["action"], 0.0)
            self.assertEqual(run_config["inference_safety"]["legality_source"], "board.gomoku.GomokuBoard")

            preview = json.loads((output_dir / "dataset_preview.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(preview["target_text"], "<MOVE_112> <EOS>")
            self.assertNotIn("robot_full_before", preview["input"]["images"])
            self.assertIn("embedding", (output_dir / "training_prompt.md").read_text(encoding="utf-8"))

    def test_adapter_formats_stage_targets_without_openvla_imports(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text(json.dumps(_manifest_record()) + "\n", encoding="utf-8")

            move_only = OpenVLAFinetuningManifestAdapter(manifest, stage="move_only")[0]
            self.assertEqual(move_only.target_text, "<MOVE_112> <EOS>")
            self.assertIsNone(move_only.teacher_move_token)

            teacher = OpenVLAFinetuningManifestAdapter(manifest, stage="teacher_move_then_action")[0]
            self.assertEqual(teacher.target_text, "<ACT_SO101_0000> <ACT_SO101_0001> <EOS>")
            self.assertEqual(teacher.teacher_move_token, "<MOVE_112>")

            full = OpenVLAFinetuningManifestAdapter(manifest, stage="move_then_action_tokens")[0]
            self.assertEqual(full.target_text, "<MOVE_112> <ACT_SO101_0000> <ACT_SO101_0001> <EOS>")

    def test_validation_rejects_input_leakage(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            record = _manifest_record()
            record["input"]["state"]["selected_move"] = [7, 7]
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "target leakage"):
                prepare_openvla_finetuning_package(manifest, Path(tmpdir) / "prep", base_model="openvla")

    def test_move_tokens_tokenizer_hook_and_legality_mask(self) -> None:
        self.assertEqual(len(build_move_tokens()), 225)
        self.assertEqual(action_index_to_move_token(112), "<MOVE_112>")

        tokenizer = _FakeTokenizer()
        added = apply_special_tokens_to_tokenizer(
            tokenizer,
            {
                "move_tokens": ["<MOVE_000>", "<MOVE_001>"],
                "action_tokens": ["<ACT_SO101_0000>"],
                "eos_token": "<EOS>",
            },
        )
        self.assertEqual(added, 4)
        self.assertIn("<MOVE_001>", tokenizer.tokens)

        board = [0] * 225
        board[112] = 1
        mask = legal_move_token_mask(board, current_player_value=2)
        self.assertFalse(mask[112])
        self.assertTrue(mask[111])

        logits = [0.0] * 225
        logits[112] = 99.0
        masked = mask_move_logits(logits, board, current_player_value=2, masked_value=-1e9)
        self.assertEqual(masked[112], -1e9)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.tokens:
                self.tokens.append(token)
                added += 1
        return added


def _manifest_record() -> dict:
    return {
        "format": "gomoku_openvla_oft_multiview_v1",
        "sample_id": "game-1_step_0000",
        "input": {
            "language_instruction": "play the strongest legal Gomoku move as black",
            "image_order": ["board_top_before", "wrist_cam_before"],
            "images": {
                "board_top_before": "samples/game-1_step_0000/inputs/board_top_before.png",
                "wrist_cam_before": "samples/game-1_step_0000/inputs/wrist_cam_before.png",
            },
            "state": {
                "board_flat": [0] * 225,
                "current_player_value": 1,
                "robot_model": "so101",
            },
        },
        "target": {
            "format": "autoregressive_move_then_action_v1",
            "text": "<MOVE_112> <ACT_SO101_0000> <ACT_SO101_0001> <EOS>",
            "tokens": ["<MOVE_112>", "<ACT_SO101_0000>", "<ACT_SO101_0001>", "<EOS>"],
            "move_token": "<MOVE_112>",
            "move": [7, 7],
            "action_index": 112,
            "policy_probs": [],
            "value": 0.1,
            "action": {
                "type": "so101_joint_trajectory",
                "tokenization": "so101_joint_tokens_v1",
                "names": ["shoulder_pan", "gripper"],
                "sequence": [[0.1, 1.0]],
                "controller_type": "so101_joint_trajectory_v1",
            },
        },
        "quality": {
            "legal": True,
            "training_usable": True,
            "execution_success": True,
            "safety_ok": True,
            "grasp_report": {"ok": True},
        },
        "qa": {
            "images": {
                "robot_full_before": "samples/game-1_step_0000/qa/robot_full_before.png",
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
