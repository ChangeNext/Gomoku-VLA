import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from board import GomokuBoard
from gomoku_ai.inference import MovePrediction
from scripts.generate_mujoco_policy_episodes import _apply_random_prefix
from simulation import GomokuMujocoEnv, collect_mujoco_policy_episode, default_mujoco_episode_output_path


class FirstLegalPredictor:
    def predict(self, board: GomokuBoard) -> MovePrediction:
        row, col = board.legal_moves()[0]
        policy = np.zeros(board.size * board.size, dtype=np.float32)
        action_index = row * board.size + col
        policy[action_index] = 1.0
        return MovePrediction(row=row, col=col, action_index=action_index, policy=policy, value=0.0)


class CenterPredictor:
    def predict(self, board: GomokuBoard) -> MovePrediction:
        row, col = board.size // 2, board.size // 2
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
            env = GomokuMujocoEnv(board_size=3, win_length=3, show_robot=True, robot_model="kinematic")

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
            self.assertTrue(record["robot_action"]["safety"]["ok"])
            self.assertEqual(record["robot_action"]["supply_before"]["black"], 5)
            self.assertEqual(record["robot_action"]["supply_after"]["black"], 4)
            self.assertIsNone(record["robot_action"]["held_stone_before"])
            self.assertIsNone(record["robot_action"]["held_stone_after"])
            self.assertEqual(record["robot_action"]["attachment_mode"], "scripted_held_stone")
            self.assertEqual(len(record["robot_action"]["action"][0]), 8)
            self.assertIn("language_instruction", record["observation"])
            self.assertEqual(record["observation"]["image_metadata"]["width"], 96)
            self.assertEqual(record["observation"]["image_metadata"]["height"], 96)
            self.assertFalse(record["observation"]["image_metadata"]["training_usable"])
            self.assertFalse(record["observation"]["image_metadata"]["phase_images_enabled"])
            self.assertEqual(record["observation"]["phase_images"], [])
            self.assertEqual(
                record["observation"]["language_instruction"],
                "play the strongest legal Gomoku move as black",
            )
            self.assertNotIn("target_cell", record["observation"]["model_input"]["state"])
            self.assertNotIn("target_world_xyz", record["observation"]["model_input"]["state"])
            self.assertIn("top_before", record["observation"]["model_input"]["images"])
            self.assertNotIn("top_after", record["observation"]["model_input"]["images"])
            self.assertEqual(record["observation"]["state"]["board_flat"][0], 0)
            self.assertEqual(record["observation"]["supervision"]["strategy"]["selected_move"], [0, 0])
            self.assertEqual(record["observation"]["supervision"]["strategy"]["board_after_flat"][0], 1)
            target_sequence = record["observation"]["supervision"]["target_sequence"]
            self.assertEqual(target_sequence["format"], "autoregressive_move_then_action_v1")
            self.assertEqual(target_sequence["move_token"], "<MOVE_000>")
            self.assertEqual(target_sequence["tokens"][0], "<MOVE_000>")
            self.assertEqual(target_sequence["tokens"][1], "<ACT_HOME>")
            self.assertEqual(target_sequence["tokens"][-1], "<EOS>")
            self.assertEqual(target_sequence["action_tokenization"], "scripted_phase_v1")
            self.assertEqual(
                target_sequence["continuous_action_source"],
                "supervision.execution.action.sequence",
            )
            self.assertEqual(record["observation"]["supervision"]["execution"]["supply_after"]["black"], 4)
            self.assertEqual(record["observation"]["supervision"]["execution"]["target_cell"], [0, 0])
            self.assertTrue(record["observation"]["state"]["safety"]["ok"])
            self.assertEqual(record["observation"]["prefix_moves"], 0)
            self.assertEqual(record["observation"]["board_ply_before"], 0)
            self.assertEqual(record["observation"]["board_ply_after"], 1)
            self.assertTrue(record["observation"]["is_first"])
            self.assertTrue(record["observation"]["is_first_recorded_frame"])
            self.assertEqual(
                record["observation"]["supervision"]["execution"]["action"]["attachment_mode"],
                "scripted_held_stone",
            )
            self.assertTrue(record["observation"]["is_last"])
            self.assertFalse(record["observation"]["is_terminal"])
            self.assertTrue(Path(record["observation"]["images"]["top_before"]).exists())
            self.assertTrue(Path(record["observation"]["images"]["top_after"]).exists())

    def test_collect_mujoco_policy_episode_can_capture_phase_images(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "episodes.jsonl"
            assets = Path(tmpdir) / "assets"
            env = GomokuMujocoEnv(board_size=3, win_length=3, show_robot=True, robot_model="kinematic")

            records = collect_mujoco_policy_episode(
                env,
                CenterPredictor(),
                output,
                assets,
                game_id="mujoco-game",
                max_moves=1,
                cameras=("top",),
                image_width=96,
                image_height=96,
                capture_phase_images=True,
            )

            self.assertEqual(len(records), 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())
            phase_images = record["observation"]["phase_images"]
            self.assertEqual(len(phase_images), len(record["robot_action"]["ee_trajectory"]))
            self.assertEqual(phase_images[0]["phase"], "home")
            self.assertEqual(phase_images[-1]["phase"], "retreat")
            self.assertTrue(record["observation"]["image_metadata"]["phase_images_enabled"])
            self.assertEqual(record["observation"]["image_metadata"]["phase_image_mode"], "scripted_visual_attachment")
            for phase_record in phase_images:
                self.assertTrue(Path(phase_record["images"]["top_phase_%03d_%s" % (
                    phase_record["index"],
                    phase_record["phase"],
                )]).exists())

    def test_collect_mujoco_policy_episode_uses_so101_joint_trajectory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "episodes.jsonl"
            assets = Path(tmpdir) / "assets"
            env = GomokuMujocoEnv(board_size=3, win_length=3, show_robot=True, robot_model="so101")

            records = collect_mujoco_policy_episode(
                env,
                CenterPredictor(),
                output,
                assets,
                game_id="mujoco-game",
                max_moves=1,
                cameras=("board_top", "wrist_cam", "robot_full"),
                image_width=96,
                image_height=96,
                capture_phase_images=True,
            )

            self.assertEqual(len(records), 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(record["robot_action"]["controller_type"], "so101_joint_trajectory_v1")
            self.assertEqual(record["robot_action"]["attachment_mode"], "constraint_style_active_stone")
            self.assertEqual(record["robot_action"]["action_names"], [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ])
            self.assertTrue(record["robot_action"]["execution_success"])
            self.assertTrue(record["robot_action"]["grasp_report"]["ok"])
            self.assertEqual(record["robot_action"]["grasp_report"]["supply_container"], "black_bowl")
            self.assertTrue(record["robot_action"]["grasp_report"]["pick_source_outside_board"])
            self.assertEqual(len(record["robot_action"]["grasp_report"]["pick_source_world_xyz"]), 3)
            self.assertEqual(len(record["robot_action"]["action"][0]), 6)
            self.assertEqual(
                record["observation"]["image_metadata"]["phase_image_mode"],
                "so101_joint_trajectory_active_stone",
            )
            self.assertEqual(record["observation"]["image_metadata"]["model_input_cameras"], ["board_top", "wrist_cam"])
            self.assertEqual(record["observation"]["image_metadata"]["qa_cameras"], ["robot_full"])
            self.assertIn("board_top_before", record["observation"]["model_input"]["images"])
            self.assertIn("wrist_cam_before", record["observation"]["model_input"]["images"])
            self.assertNotIn("robot_full_before", record["observation"]["model_input"]["images"])
            self.assertIn("robot_full_before", record["observation"]["images"])
            self.assertIn("wrist_cam_before", record["observation"]["images"])
            self.assertIn("wrist_cam_phase_002_pick", record["observation"]["phase_images"][2]["images"])
            self.assertTrue(Path(record["observation"]["image_metadata"]["qa_contact_sheet"]).exists())
            target_sequence = record["observation"]["supervision"]["target_sequence"]
            self.assertEqual(target_sequence["action_tokenization"], "so101_joint_tokens_v1")
            self.assertTrue(target_sequence["action_tokens"][0].startswith("<ACT_SO101_"))

    def test_random_prefix_updates_board_visuals_and_supply(self) -> None:
        import random

        env = GomokuMujocoEnv(board_size=5, win_length=4, rule_set="free", enforce_center_opening=False)

        applied = _apply_random_prefix(env, 3, random.Random(7))

        self.assertEqual(applied, 3)
        self.assertEqual(env.board.move_count, 3)
        self.assertEqual(env.supply_counts[env.board.current_player.opponent], 11)
        self.assertEqual(env.supply_counts[env.board.current_player], 11)
        occupied = [
            (row, col)
            for row in range(env.board.size)
            for col in range(env.board.size)
            if env.board.grid[row][col] != 0
        ]
        for row, col in occupied:
            geom_id = env.model.geom(f"stone_{row}_{col}").id
            self.assertGreater(float(env.model.geom_rgba[geom_id][3]), 0.0)

    def test_prefix_collection_marks_recorded_frame_separately_from_game_first_move(self) -> None:
        import random

        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "episodes.jsonl"
            assets = Path(tmpdir) / "assets"
            env = GomokuMujocoEnv(board_size=5, win_length=4, rule_set="free", enforce_center_opening=False)

            applied = _apply_random_prefix(env, 3, random.Random(11))
            records = collect_mujoco_policy_episode(
                env,
                FirstLegalPredictor(),
                output,
                assets,
                game_id="prefixed-game",
                max_moves=env.board.move_count + 1,
                cameras=("top",),
                image_width=96,
                image_height=96,
            )

            self.assertEqual(applied, 3)
            self.assertEqual(len(records), 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())
            observation = record["observation"]
            self.assertEqual(observation["prefix_moves"], 3)
            self.assertEqual(observation["board_ply_before"], 3)
            self.assertEqual(observation["board_ply_after"], 4)
            self.assertFalse(observation["is_first"])
            self.assertTrue(observation["is_first_recorded_frame"])


if __name__ == "__main__":
    unittest.main()
