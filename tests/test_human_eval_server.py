import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from board import GomokuBoard
from scripts.human_eval_server import HumanEvalConfig, HumanEvalStore, resolve_output_paths


class FirstLegalSelector:
    board_size = 3
    rule_set = "free"
    enforce_center_opening = False

    def __call__(self, board: GomokuBoard) -> tuple[int, int]:
        return board.legal_moves()[0]


class HumanEvalServerTest(unittest.TestCase):
    def test_default_output_paths_use_checkpoint_run_evaluation_dir(self) -> None:
        jsonl_path, csv_path = resolve_output_paths(
            HumanEvalConfig(checkpoint="gomoku_ai/runs/example/checkpoints/best.pt")
        )

        self.assertEqual(jsonl_path, Path("gomoku_ai/runs/example/evaluation/best_human_eval.jsonl"))
        self.assertEqual(csv_path, Path("gomoku_ai/runs/example/evaluation/best_human_eval.csv"))

    def test_new_game_records_ai_opening_when_human_is_white(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = HumanEvalStore(
                HumanEvalConfig(
                    checkpoint="best.pt",
                    output_jsonl=str(Path(tmpdir) / "games.jsonl"),
                    output_csv=str(Path(tmpdir) / "games.csv"),
                    win_length=3,
                ),
                FirstLegalSelector(),
            )

            state = store.new_game("tester", "white")

            self.assertEqual(state["human_color"], "white")
            self.assertEqual(state["move_count"], 1)
            self.assertEqual(state["moves"][0], {"player": "black", "row": 0, "col": 0})

    def test_completed_game_is_written_to_jsonl_and_stats(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "games.jsonl"
            store = HumanEvalStore(
                HumanEvalConfig(
                    checkpoint="best.pt",
                    output_jsonl=str(output),
                    output_csv=str(Path(tmpdir) / "games.csv"),
                    win_length=3,
                ),
                FirstLegalSelector(),
            )
            state = store.new_game("tester", "black")

            for row, col in ((1, 0), (1, 1), (1, 2)):
                state = store.human_move(state["game_id"], row, col)
                if state["result"] is not None:
                    break

            self.assertEqual(state["result"], "human_win")
            record = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(record["player_id"], "tester")
            self.assertEqual(record["result"], "human_win")
            self.assertEqual(record["human_color"], "black")
            self.assertEqual(store.stats()["totals"]["human_win"], 1)
            self.assertIn("human_win", (Path(tmpdir) / "games.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
