import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from board import GomokuBoard
from gomoku_ai.episode_recorder import default_episode_output_path, play_and_record_episode
from gomoku_ai.inference import MovePrediction


class FirstLegalPredictor:
    def predict(self, board: GomokuBoard) -> MovePrediction:
        row, col = board.legal_moves()[0]
        policy = np.zeros(board.size * board.size, dtype=np.float32)
        action_index = row * board.size + col
        policy[action_index] = 1.0
        return MovePrediction(
            row=row,
            col=col,
            action_index=action_index,
            policy=policy,
            value=0.25,
        )


class IllegalPredictor:
    def predict(self, board: GomokuBoard) -> MovePrediction:
        policy = np.zeros(board.size * board.size, dtype=np.float32)
        policy[0] = 1.0
        return MovePrediction(row=0, col=0, action_index=0, policy=policy, value=0.0)


class EpisodeRecorderTest(unittest.TestCase):
    def test_default_episode_output_path_uses_checkpoint_run_data_dir(self) -> None:
        path = default_episode_output_path("gomoku_ai/runs/example/checkpoints/best.pt")

        self.assertEqual(path, Path("gomoku_ai/runs/example/data/best_policy_episodes.jsonl"))

    def test_play_and_record_episode_writes_move_level_jsonl(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "episodes.jsonl"
            board = GomokuBoard(size=3, win_length=3)

            records = play_and_record_episode(
                board,
                FirstLegalPredictor(),
                output,
                game_id="game-1",
                checkpoint="best.pt",
                max_moves=2,
            )

            self.assertEqual(len(records), 2)
            lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["game_id"], "game-1")
            self.assertEqual(lines[0]["step"], 0)
            self.assertEqual(lines[0]["board_before"], [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.assertEqual(lines[0]["selected_move"], [0, 0])
            self.assertEqual(lines[0]["board_after"][0][0], 1)
            self.assertEqual(lines[0]["current_player"], "black")
            self.assertEqual(lines[0]["policy_source"], "alphazero")
            self.assertEqual(lines[0]["checkpoint"], "best.pt")
            self.assertEqual(len(lines[0]["policy_probs"]), 9)
            self.assertTrue(lines[0]["legal"])
            self.assertFalse(lines[0]["terminal"])

    def test_play_and_record_episode_records_illegal_move_and_stops(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "episodes.jsonl"
            board = GomokuBoard(size=3, win_length=3)
            board.place(0, 0)

            records = play_and_record_episode(board, IllegalPredictor(), output, max_moves=3)

            self.assertEqual(len(records), 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertFalse(record["legal"])
            self.assertTrue(record["terminal"])
            self.assertIn("illegal move", record["error"])
            self.assertEqual(board.move_count, 1)


if __name__ == "__main__":
    unittest.main()
