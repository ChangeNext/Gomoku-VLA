import unittest

from scripts.validate_so101_dataset import _find_forbidden_keys


class OpenVLAManifestValidationTest(unittest.TestCase):
    def test_input_contract_allows_only_pre_action_fields(self) -> None:
        model_input = {
            "language_instruction": "play the strongest legal Gomoku move as black",
            "images": {"board_top_before": "a.png", "wrist_cam_before": "b.png"},
            "state": {"board_flat": [0], "current_player_value": 1},
        }
        self.assertEqual(list(_find_forbidden_keys(model_input)), [])

    def test_input_contract_rejects_supervision(self) -> None:
        self.assertEqual(list(_find_forbidden_keys({"state": {"action_index": 4}})), ["action_index"])


if __name__ == "__main__":
    unittest.main()
