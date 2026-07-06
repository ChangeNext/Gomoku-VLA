import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from board import GomokuBoard
from gomoku_ai.inference import MovePrediction
from simulation import GomokuMujocoEnv, collect_mujoco_policy_episode, default_mujoco_episode_output_path


class FirstLegalPredictor:
    def predict(self, board: GomokuBoard) -> MovePrediction:
        row, col = board.legal_moves()[0]
        policy = np.zeros(board.size * board.size, dtype=np.float32)
        action_index = row * board.size + col
        policy[action_index] = 1.0
        return MovePrediction(row=row, col=col, action_index=action_index, policy=policy, value=0.0)


class MujocoPolicyCollectionTest(unittest.TestCase):
    def test_default_mujoco_episode_output_path_uses_checkpoint_run_data_dir(self) -> None:
        path = default_mujoco_episode_output_path("gomoku_ai/runs/example/checkpoints/best.pt")

        self.assertEqual(path, Path("gomoku_ai/runs/example/data/best_mujoco_policy_episodes.jsonl"))

    def test_collect_mujoco_policy_episode_writes_images_actions_and_jsonl(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "episodes.jsonl"
            assets = Path(tmpdir) / "assets"
            env = GomokuMujocoEnv(board_size=3, win_length=3, show_robot=True)

            records = collect_mujoco_policy_episode(
                env,
                FirstLegalPredictor(),
                output,
                assets,
                game_id="mujoco-game",
                max_moves=1,
                cameras=("top",),
                image_width=96,
                image_height=96,
            )

            self.assertEqual(len(records), 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(record["game_id"], "mujoco-game")
            self.assertEqual(record["selected_move"], [0, 0])
            self.assertEqual(record["board_after"][0][0], 1)
            self.assertEqual(record["robot_action"]["controller_type"], "scripted_kinematic_v1")
            self.assertTrue(record["robot_action"]["execution_success"])
            self.assertEqual(len(record["robot_action"]["action"][0]), 8)
            self.assertIn("language_instruction", record["observation"])
            self.assertEqual(record["observation"]["state"]["target_cell"], [0, 0])
            self.assertEqual(record["observation"]["state"]["board_flat"][0], 0)
            self.assertTrue(record["observation"]["is_last"])
            self.assertFalse(record["observation"]["is_terminal"])
            self.assertTrue(Path(record["observation"]["images"]["top_before"]).exists())
            self.assertTrue(Path(record["observation"]["images"]["top_after"]).exists())


if __name__ == "__main__":
    unittest.main()
