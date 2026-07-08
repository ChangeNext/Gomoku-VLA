import unittest

from board import Player
from robot_control import RobotSafetyController
from simulation import GomokuMujocoEnv, build_pick_place_action


class RobotSafetyControllerTest(unittest.TestCase):
    def test_accepts_legal_pick_and_place(self) -> None:
        env = GomokuMujocoEnv()
        safety = RobotSafetyController(env)

        self.assertTrue(safety.validate_pick(Player.BLACK).ok)
        self.assertTrue(safety.validate_place_cell(7, 7, Player.BLACK).ok)
        self.assertTrue(safety.validate_action_trace(build_pick_place_action(env, (7, 7), Player.BLACK)).ok)

    def test_rejects_occupied_target(self) -> None:
        env = GomokuMujocoEnv()
        env.step((7, 7))
        safety = RobotSafetyController(env)

        report = safety.validate_place_cell(7, 7)

        self.assertFalse(report.ok)
        self.assertIn("illegal", report.reason or "")

    def test_grasp_supply_decrements_inventory_and_commit_places_stone(self) -> None:
        env = GomokuMujocoEnv()
        before = env.supply_counts[Player.BLACK]

        env.grasp_supply_stone(Player.BLACK)
        self.assertEqual(env.held_stone_player, Player.BLACK)
        self.assertEqual(env.supply_counts[Player.BLACK], before - 1)

        env.commit_held_stone_to_cell(7, 7)
        self.assertIsNone(env.held_stone_player)
        self.assertEqual(env.board.grid[7][7], Player.BLACK.value)


if __name__ == "__main__":
    unittest.main()
