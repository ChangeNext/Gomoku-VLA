import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gomoku_ai.openvla_manifest_split import split_openvla_manifest_by_game


class OpenVLAManifestSplitTest(unittest.TestCase):
    def test_split_manifest_groups_by_game_id(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.jsonl"
            records = []
            for game_index in range(10):
                for step in range(game_index % 3 + 1):
                    records.append(_record(game_id=f"game-{game_index}", step=step))
            manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            summary = split_openvla_manifest_by_game(
                manifest,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=7,
            )

            self.assertEqual(summary.total_samples, len(records))
            self.assertEqual(summary.total_games, 10)
            self.assertEqual(summary.split_games, {"train": 6, "val": 2, "test": 2})
            train = _read(root / "manifest_train.jsonl")
            val = _read(root / "manifest_val.jsonl")
            test = _read(root / "manifest_test.jsonl")
            self.assertEqual(len(train) + len(val) + len(test), len(records))
            train_games = _game_ids(train)
            val_games = _game_ids(val)
            test_games = _game_ids(test)
            self.assertFalse(train_games & val_games)
            self.assertFalse(train_games & test_games)
            self.assertFalse(val_games & test_games)
            metadata = json.loads((root / "manifest_splits.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["policy"], "game_id_grouped_shuffle")

    def test_split_requires_game_id(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text(json.dumps({"sample_id": "missing-source"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source.game_id"):
                split_openvla_manifest_by_game(manifest)


def _record(*, game_id: str, step: int) -> dict:
    return {
        "sample_id": f"{game_id}-{step}",
        "source": {"game_id": game_id, "step": step},
        "input": {},
        "target": {},
    }


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _game_ids(records: list[dict]) -> set[str]:
    return {record["source"]["game_id"] for record in records}


if __name__ == "__main__":
    unittest.main()
