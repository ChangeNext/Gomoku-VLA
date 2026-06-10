import unittest

from simulation import GomokuMujocoEnv


class GomokuMujocoEnvTest(unittest.TestCase):
    def test_places_stone_and_rebuilds_model(self) -> None:
        env = GomokuMujocoEnv()
        env.step((7, 7))
        self.assertEqual(env.board.grid[7][7], 1)
        self.assertGreater(env.model.ngeom, 0)
        self.assertEqual(env.robot_target_cell, (7, 7))

    def test_human_move_can_skip_robot_target_update(self) -> None:
        env = GomokuMujocoEnv()
        env.step((7, 7), update_robot_target=True)
        env.step((7, 8), update_robot_target=False)
        self.assertEqual(env.board.grid[7][8], 2)
        self.assertEqual(env.robot_target_cell, (7, 7))

    def test_first_human_move_keeps_robot_without_target(self) -> None:
        env = GomokuMujocoEnv()
        env.step((7, 7), update_robot_target=False)
        self.assertEqual(env.board.grid[7][7], 1)
        self.assertIsNone(env.robot_target_cell)

    def test_coordinate_round_trip(self) -> None:
        env = GomokuMujocoEnv()
        x, y, _ = env.board_to_world(3, 11)
        self.assertEqual(env.world_to_board(x, y), (3, 11))

    def test_robot_geoms_are_in_model(self) -> None:
        env = GomokuMujocoEnv()
        names = {env.model.geom(i).name for i in range(env.model.ngeom)}
        self.assertIn("panda_link3", names)
        self.assertIn("panda_finger_left", names)


if __name__ == "__main__":
    unittest.main()
