import unittest

from board import Player
from simulation import GomokuMujocoEnv, build_pick_place_action


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

    def test_can_set_selection_directly(self) -> None:
        env = GomokuMujocoEnv()
        self.assertEqual(env.set_selection(4, 5), (4, 5))
        self.assertEqual(env.selected_cell, (4, 5))

    def test_stone_supply_positions_are_outside_board_and_player_specific(self) -> None:
        env = GomokuMujocoEnv()
        black_supply = env.stone_supply_world(Player.BLACK)
        white_supply = env.stone_supply_world(Player.WHITE)
        self.assertNotEqual(black_supply, white_supply)
        with self.assertRaises(ValueError):
            env.world_to_board(black_supply[0], black_supply[1])

    def test_scripted_pick_place_action_has_lerobot_ready_action_vectors(self) -> None:
        env = GomokuMujocoEnv()
        action = build_pick_place_action(env, (7, 7), Player.BLACK)
        phases = [point["phase"] for point in action["ee_trajectory"]]
        self.assertEqual(
            phases,
            ["home", "pre_pick", "pick", "grasp", "lift", "pre_place", "place", "release", "retreat"],
        )
        self.assertEqual(action["controller_type"], "scripted_kinematic_v1")
        self.assertEqual(action["target_cell"], [7, 7])
        self.assertEqual(action["joint_trajectory"], None)
        self.assertTrue(all(len(vector) == 8 for vector in action["action"]))
        self.assertEqual(action["action_names"], ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"])

    def test_can_move_robot_hand_to_scripted_pose(self) -> None:
        env = GomokuMujocoEnv()
        env.set_robot_hand_world(0.1, -0.2, 0.08, gripper=0.0)
        hand_id = env.model.geom("panda_hand").id
        left_id = env.model.geom("panda_finger_left").id
        right_id = env.model.geom("panda_finger_right").id

        self.assertAlmostEqual(float(env.model.geom_pos[hand_id][0]), 0.1)
        self.assertAlmostEqual(float(env.model.geom_pos[hand_id][1]), -0.2)
        self.assertAlmostEqual(float(env.model.geom_pos[hand_id][2]), 0.08)
        self.assertLess(float(env.model.geom_pos[left_id][0]), 0.1)
        self.assertGreater(float(env.model.geom_pos[right_id][0]), 0.1)


if __name__ == "__main__":
    unittest.main()
