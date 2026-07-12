import unittest
from math import dist

import numpy as np

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
        self.assertIn("cursor", names)
        self.assertIn("held_stone", names)
        self.assertIn("board", names)

    def test_menagerie_panda_loads_with_joints_and_sites(self) -> None:
        env = GomokuMujocoEnv(robot_model="panda")
        joint_names = {env.model.joint(i).name for i in range(env.model.njnt)}
        site_names = {env.model.site(i).name for i in range(env.model.nsite)}

        for name in env.panda_joint_names:
            self.assertIn(name, joint_names)
        for name in env.panda_site_names:
            self.assertIn(name, site_names)
        self.assertEqual(env.model.nu, 8)

    def test_menagerie_panda_ik_moves_above_target_cell(self) -> None:
        env = GomokuMujocoEnv(robot_model="panda")
        target = env.panda_target_pose_for_cell(7, 7)
        joint_targets = env.solve_panda_ik(target)

        self.assertEqual(len(joint_targets), 7)
        env.set_panda_joint_targets(joint_targets, gripper=1.0)
        env.simulate(3000)

        self.assertLess(dist(target, env.panda_ee_world()), 0.015)

    def test_menagerie_panda_gripper_control_sets_actuator(self) -> None:
        env = GomokuMujocoEnv(robot_model="panda")
        env.set_panda_gripper(0.0)
        self.assertAlmostEqual(float(env.data.ctrl[7]), 0.0)
        env.set_panda_gripper(1.0)
        self.assertAlmostEqual(float(env.data.ctrl[7]), 255.0)

    def test_menagerie_so101_loads_with_joints_and_sites(self) -> None:
        env = GomokuMujocoEnv(robot_model="so101")
        joint_names = {env.model.joint(i).name for i in range(env.model.njnt)}
        site_names = {env.model.site(i).name for i in range(env.model.nsite)}

        for name in env.so101_joint_names:
            self.assertIn(name, joint_names)
        for name in env.so101_site_names:
            self.assertIn(name, site_names)
        self.assertEqual(env.model.nu, 6)

    def test_menagerie_so101_base_starts_outside_play_grid(self) -> None:
        env = GomokuMujocoEnv(robot_model="so101")
        base_x = float(env.model.body("base").pos[0])

        self.assertGreater(base_x, env.board_extent / 2.0 + env.cell_size * 3.0)

    def test_menagerie_so101_ik_moves_above_target_cell(self) -> None:
        env = GomokuMujocoEnv(robot_model="so101")
        target = env.so101_target_pose_for_cell(7, 7)
        joint_targets = env.solve_so101_ik(target)

        self.assertEqual(len(joint_targets), 5)
        env.set_so101_joint_targets(joint_targets, gripper=1.0)
        env.simulate(1200)

        self.assertLess(dist(target, env.so101_ee_world()), 0.035)

    def test_menagerie_so101_ik_keeps_gripper_vertical_over_cell(self) -> None:
        import mujoco

        env = GomokuMujocoEnv(robot_model="so101")
        joint_targets = env.solve_so101_ik(env.so101_target_pose_for_cell(7, 7))
        for index, addr in enumerate(env._so101_arm_qpos_addrs()):
            env.data.qpos[addr] = joint_targets[index]
        mujoco.mj_forward(env.model, env.data)

        site_id = env.model.site("so101_ee_site").id
        site_z = np.array(env.data.site_xmat[site_id], dtype=float).reshape(3, 3)[:, 2]
        self.assertGreater(float(site_z[2]), 0.96)

    def test_robot_scale_board_keeps_all_corners_within_so101_place_tolerance(self) -> None:
        env = GomokuMujocoEnv(cell_size=0.021, stone_radius=0.007, robot_model="so101")
        for row, col in ((0, 0), (0, 14), (14, 0), (14, 14)):
            target = np.array(env.board_to_world(row, col), dtype=float)
            ik_target = (float(target[0]), float(target[1]), float(target[2] + 0.014))
            env.set_so101_joint_targets(env.solve_so101_ik(ik_target), gripper=0.0)
            carried = np.array(env.so101_gripper_world(), dtype=float) + np.array([0.0, 0.0, -0.014])
            self.assertLess(float(np.linalg.norm(carried - target)), 0.055)

    def test_articulated_robot_cursor_color_is_neutral(self) -> None:
        env = GomokuMujocoEnv(robot_model="so101")
        cursor_id = env.model.geom("cursor").id
        initial_rgba = env.model.geom_rgba[cursor_id].copy()

        env.set_selection(7, 8)
        moved_rgba = env.model.geom_rgba[cursor_id]

        self.assertTrue(np.allclose(initial_rgba, moved_rgba))

    def test_menagerie_so101_gripper_control_sets_actuator(self) -> None:
        env = GomokuMujocoEnv(robot_model="so101")
        gripper_id = env.model.actuator("gripper").id
        low, high = env.model.actuator_ctrlrange[gripper_id]
        env.set_so101_gripper(0.0)
        self.assertAlmostEqual(float(env.data.ctrl[gripper_id]), float(low))
        env.set_so101_gripper(1.0)
        self.assertAlmostEqual(float(env.data.ctrl[gripper_id]), float(high))

    def test_can_set_selection_directly(self) -> None:
        env = GomokuMujocoEnv()
        self.assertEqual(env.set_selection(4, 5), (4, 5))
        self.assertEqual(env.selected_cell, (4, 5))

    def test_stone_supply_positions_are_outside_board_and_player_specific(self) -> None:
        env = GomokuMujocoEnv()
        black_supply = env.stone_supply_world(Player.BLACK)
        white_supply = env.stone_supply_world(Player.WHITE)
        self.assertNotEqual(black_supply, white_supply)
        board_half = env.board_extent / 2.0 + env.cell_size * 0.7
        self.assertGreater(abs(black_supply[0]), board_half)
        self.assertGreater(abs(white_supply[0]), board_half)
        with self.assertRaises(ValueError):
            env.world_to_board(black_supply[0], black_supply[1])

    def test_scene_contains_black_and_white_stone_bowls(self) -> None:
        env = GomokuMujocoEnv()

        for name in (
            "black_bowl_base",
            "black_bowl_inner",
            "black_bowl_rim",
            "black_bowl_stone_0",
            "white_bowl_base",
            "white_bowl_inner",
            "white_bowl_rim",
            "white_bowl_stone_0",
        ):
            self.assertGreaterEqual(env.model.geom(name).id, 0)

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
        env = GomokuMujocoEnv(robot_model="kinematic")
        env.set_robot_hand_world(0.1, -0.2, 0.08, gripper=0.0)
        hand_id = env.model.geom("panda_hand").id
        left_id = env.model.geom("panda_finger_left").id
        right_id = env.model.geom("panda_finger_right").id

        self.assertAlmostEqual(float(env.model.geom_pos[hand_id][0]), 0.1)
        self.assertAlmostEqual(float(env.model.geom_pos[hand_id][1]), -0.2)
        self.assertAlmostEqual(float(env.model.geom_pos[hand_id][2]), 0.08)
        self.assertLess(float(env.model.geom_pos[left_id][0]), 0.1)
        self.assertGreater(float(env.model.geom_pos[right_id][0]), 0.1)

    def test_held_stone_can_be_shown_and_cleared(self) -> None:
        env = GomokuMujocoEnv()
        held_id = env.model.geom("held_stone").id

        self.assertAlmostEqual(float(env.model.geom_rgba[held_id][3]), 0.0)
        env.set_held_stone_world(0.1, -0.2, 0.08, Player.WHITE)

        self.assertAlmostEqual(float(env.model.geom_pos[held_id][0]), 0.1)
        self.assertAlmostEqual(float(env.model.geom_pos[held_id][1]), -0.2)
        self.assertAlmostEqual(float(env.model.geom_pos[held_id][2]), 0.08)
        self.assertAlmostEqual(float(env.model.geom_rgba[held_id][3]), 1.0)
        self.assertGreater(float(env.model.geom_rgba[held_id][0]), 0.8)

        env.clear_held_stone()
        self.assertAlmostEqual(float(env.model.geom_rgba[held_id][3]), 0.0)


if __name__ == "__main__":
    unittest.main()
