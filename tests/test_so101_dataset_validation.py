import unittest

from scripts.validate_so101_dataset import _find_forbidden_keys, _valid_action_sequence, _valid_board_transition


class SO101DatasetValidationTest(unittest.TestCase):
    def test_accepts_exact_selected_move_transition(self) -> None:
        before = [[0, 0], [0, 0]]
        after = [[0, 0], [1, 0]]
        self.assertTrue(_valid_board_transition(before, after, [1, 0], 1))

    def test_rejects_transition_with_two_changed_cells(self) -> None:
        before = [[0, 0], [0, 0]]
        after = [[2, 0], [1, 0]]
        self.assertFalse(_valid_board_transition(before, after, [1, 0], 1))

    def test_rejects_non_finite_or_wrong_width_actions(self) -> None:
        self.assertTrue(_valid_action_sequence(["joint", "gripper"], [[0.2, 1.0]]))
        self.assertFalse(_valid_action_sequence(["joint", "gripper"], [[0.2]]))
        self.assertFalse(_valid_action_sequence(["joint"], [[float("inf")]]))

    def test_detects_nested_target_and_future_image_leakage(self) -> None:
        value = {"state": {"target_cell": [1, 2]}, "images": {"board_top_after": "future.png"}}
        self.assertEqual(set(_find_forbidden_keys(value)), {"target_cell", "board_top_after"})

if __name__ == "__main__":
    unittest.main()
