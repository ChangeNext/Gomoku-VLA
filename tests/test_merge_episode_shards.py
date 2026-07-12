import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.merge_episode_shards import merge_shards


class MergeEpisodeShardsTest(unittest.TestCase):
    def test_merges_in_episode_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            left = root / "left.jsonl"
            right = root / "right.jsonl"
            output = root / "merged.jsonl"
            left.write_text(json.dumps({"game_id": "b", "step": 0, "observation": {"episode_index": 1}}) + "\n")
            right.write_text(json.dumps({"game_id": "a", "step": 0, "observation": {"episode_index": 0}}) + "\n")

            games, records = merge_shards([left, right], output)

            merged = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual((games, records), (2, 2))
            self.assertEqual([record["game_id"] for record in merged], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
