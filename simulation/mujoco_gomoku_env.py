from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from board import GomokuBoard, Player
from board.gomoku import RuleSet


class GomokuMujocoEnv:
    """MuJoCo-backed Gomoku scene with in-place updates for viewer use."""

    def __init__(
        self,
        board_size: int = 15,
        win_length: int = 5,
        rule_set: RuleSet = "free",
        enforce_center_opening: bool = False,
        cell_size: float = 0.035,
        stone_radius: float = 0.012,
        stone_height: float = 0.006,
        show_robot: bool = True,
        robot_model: str = "so101",
    ) -> None:
        if robot_model not in {"kinematic", "panda", "so101"}:
            raise ValueError("robot_model must be 'kinematic', 'panda', or 'so101'")
        self.board = GomokuBoard(
            size=board_size,
            win_length=win_length,
            rule_set=rule_set,
            enforce_center_opening=enforce_center_opening,
        )
        self.cell_size = cell_size
        self.stone_radius = stone_radius
        self.stone_height = stone_height
        self.show_robot = show_robot
        self.robot_model = robot_model
        self.selected_cell = (self.board.size // 2, self.board.size // 2)
        self.robot_target_cell: tuple[int, int] | None = None
        self.supply_counts = {
            Player.BLACK: (self.board.size * self.board.size + 1) // 2,
            Player.WHITE: (self.board.size * self.board.size) // 2,
        }
        self.held_stone_player: Player | None = None
        self.active_stone_player: Player | None = None
        self.active_stone_attached = False
        self._model_xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(self._model_xml)
        self.data = mujoco.MjData(self.model)
        self._apply_robot_home_keyframe()
        self._geom_ids = self._collect_geom_ids()
        self._reset_runtime_geoms()
        mujoco.mj_forward(self.model, self.data)

    @property
    def board_extent(self) -> float:
        return (self.board.size - 1) * self.cell_size

    def reset(self) -> None:
        self.board.reset()
        self.selected_cell = (self.board.size // 2, self.board.size // 2)
        self.robot_target_cell = None
        self.supply_counts = {
            Player.BLACK: (self.board.size * self.board.size + 1) // 2,
            Player.WHITE: (self.board.size * self.board.size) // 2,
        }
        self.held_stone_player = None
        self.active_stone_player = None
        self.active_stone_attached = False
        self._apply_robot_home_keyframe()
        self._reset_runtime_geoms()
        mujoco.mj_forward(self.model, self.data)

    def step(self, action: tuple[int, int], update_robot_target: bool = True) -> dict[str, object]:
        row, col = action
        winner = self.board.place(row, col)
        self.selected_cell = (row, col)
        if update_robot_target:
            self.robot_target_cell = (row, col)
        self._set_stone_geom(row, col, Player(self.board.grid[row][col]))
        self._update_cursor_geom()
        if update_robot_target:
            self._update_hand_geoms()
        mujoco.mj_forward(self.model, self.data)
        return {
            "board": self.board.copy_state(),
            "current_player": self.board.current_player,
            "winner": winner,
            "done": winner is not None,
            "move_count": self.board.move_count,
        }

    def place_selected(self, update_robot_target: bool = True) -> dict[str, object]:
        return self.step(self.selected_cell, update_robot_target=update_robot_target)

    def move_selection(self, d_row: int, d_col: int) -> tuple[int, int]:
        row, col = self.selected_cell
        row = min(max(row + d_row, 0), self.board.size - 1)
        col = min(max(col + d_col, 0), self.board.size - 1)
        self.selected_cell = (row, col)
        self._update_cursor_geom()
        mujoco.mj_forward(self.model, self.data)
        return self.selected_cell

    def set_selection(self, row: int, col: int, update_robot_target: bool = False) -> tuple[int, int]:
        if not self.board.is_on_board(row, col):
            raise ValueError(f"cell out of range: row={row}, col={col}")
        self.selected_cell = (row, col)
        if update_robot_target:
            self.robot_target_cell = (row, col)
            self._update_hand_geoms()
        self._update_cursor_geom()
        mujoco.mj_forward(self.model, self.data)
        return self.selected_cell

    def board_to_world(self, row: int, col: int) -> tuple[float, float, float]:
        if not self.board.is_on_board(row, col):
            raise ValueError(f"cell out of range: row={row}, col={col}")
        half = self.board_extent / 2.0
        x = col * self.cell_size - half
        y = half - row * self.cell_size
        z = 0.012 + self.stone_height
        return x, y, z

    def world_to_board(self, x: float, y: float) -> tuple[int, int]:
        half = self.board_extent / 2.0
        col = round((x + half) / self.cell_size)
        row = round((half - y) / self.cell_size)
        if not self.board.is_on_board(row, col):
            raise ValueError(f"world position outside board: x={x}, y={y}")
        return row, col

    def stone_supply_world(self, player: Player) -> tuple[float, float, float]:
        if player not in {Player.BLACK, Player.WHITE}:
            raise ValueError("player must be BLACK or WHITE")
        x, y = self.stone_bowl_world(player)
        z = 0.012 + self.stone_height
        return x, y, z

    def stone_bowl_world(self, player: Player) -> tuple[float, float]:
        if player not in {Player.BLACK, Player.WHITE}:
            raise ValueError("player must be BLACK or WHITE")
        half = self.board_extent / 2.0
        x = half + self.cell_size * 2.45
        y = self.cell_size * (-3.00 if player == Player.BLACK else 3.00)
        return x, y

    def robot_home_world(self) -> tuple[float, float, float]:
        return self._robot_home_world()

    def set_robot_hand_world(self, x: float, y: float, z: float, gripper: float = 1.0) -> None:
        if not self.show_robot:
            raise ValueError("robot geoms are disabled")
        if self.robot_model != "kinematic":
            raise ValueError("set_robot_hand_world is only available for robot_model='kinematic'")
        self._set_hand_geoms(x, y, z, gripper)
        mujoco.mj_forward(self.model, self.data)

    def set_held_stone_world(self, x: float, y: float, z: float, player: Player) -> None:
        if player not in {Player.BLACK, Player.WHITE}:
            raise ValueError("player must be BLACK or WHITE")
        if self.held_stone_player is not None and self.held_stone_player != player:
            raise ValueError("cannot change held stone color while carrying another stone")
        self.held_stone_player = player
        geom_id = self._geom_ids["held_stone"]
        self.model.geom_pos[geom_id] = np.array([x, y, z], dtype=float)
        self.model.geom_rgba[geom_id] = self._stone_rgba(player)
        mujoco.mj_forward(self.model, self.data)

    def clear_held_stone(self) -> None:
        self.held_stone_player = None
        geom_id = self._geom_ids["held_stone"]
        self.model.geom_rgba[geom_id] = self._stone_rgba(Player.EMPTY)
        mujoco.mj_forward(self.model, self.data)

    def set_active_stone_world(self, x: float, y: float, z: float, player: Player) -> None:
        if player not in {Player.BLACK, Player.WHITE}:
            raise ValueError("player must be BLACK or WHITE")
        joint_id = self._active_stone_joint_id()
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qpos_addr : qpos_addr + 7] = np.array([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.qvel[:] = 0.0
        self.active_stone_player = player
        geom_id = self._geom_ids["active_stone"]
        self.model.geom_rgba[geom_id] = self._stone_rgba(player)
        mujoco.mj_forward(self.model, self.data)

    def clear_active_stone(self) -> None:
        joint_id = self._active_stone_joint_id()
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qpos_addr : qpos_addr + 7] = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.qvel[:] = 0.0
        self.active_stone_player = None
        self.active_stone_attached = False
        geom_id = self._geom_ids["active_stone"]
        self.model.geom_rgba[geom_id] = self._stone_rgba(Player.EMPTY)
        mujoco.mj_forward(self.model, self.data)

    def active_stone_world(self) -> tuple[float, float, float] | None:
        if self.active_stone_player is None:
            return None
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "active_stone_body")
        if body_id < 0:
            return None
        mujoco.mj_forward(self.model, self.data)
        return tuple(float(value) for value in self.data.xpos[body_id])

    def spawn_active_stone_at_supply(self, player: Player) -> None:
        if player not in {Player.BLACK, Player.WHITE}:
            raise ValueError("player must be BLACK or WHITE")
        if self.active_stone_player is not None:
            raise ValueError("an active stone is already present")
        if self.supply_counts[player] <= 0:
            raise ValueError(f"no {player.name.lower()} stones left in supply")
        x, y, z = self.stone_supply_world(player)
        self.supply_counts[player] -= 1
        self.set_active_stone_world(x, y, z, player)

    def attach_active_stone_to_so101_gripper(self, player: Player, *, z_offset: float = -0.014) -> None:
        self._require_so101()
        if self.active_stone_player is None:
            self.spawn_active_stone_at_supply(player)
        if self.active_stone_player != player:
            raise ValueError("cannot attach a stone for a different player")
        self.active_stone_attached = True
        self.update_active_stone_at_so101_gripper(player, z_offset=z_offset)

    def update_active_stone_at_so101_gripper(self, player: Player, *, z_offset: float = -0.014) -> None:
        self._require_so101()
        if not self.active_stone_attached:
            return
        x, y, z = self.so101_gripper_world()
        self.set_active_stone_world(x, y, z + z_offset, player)
        self.active_stone_attached = True

    def release_active_stone_at(self, x: float, y: float, z: float, player: Player) -> None:
        if self.active_stone_player != player:
            raise ValueError("cannot release a missing or mismatched active stone")
        self.active_stone_attached = False
        self.set_active_stone_world(x, y, z, player)

    def commit_active_stone_to_cell(self, row: int, col: int, update_robot_target: bool = True) -> dict[str, object]:
        if self.active_stone_player is None:
            raise ValueError("robot is not carrying an active stone")
        if self.active_stone_player != self.board.current_player:
            raise ValueError(
                f"active stone is {self.active_stone_player.name.lower()}, "
                f"but current player is {self.board.current_player.name.lower()}"
            )
        stone_pos = self.active_stone_world()
        if stone_pos is None:
            raise ValueError("active stone position is unavailable")
        final_cell = self.world_to_board(stone_pos[0], stone_pos[1])
        if final_cell != (row, col):
            raise ValueError(f"active stone settled at {final_cell}, expected {(row, col)}")
        self.clear_active_stone()
        try:
            return self.step((row, col), update_robot_target=update_robot_target)
        except Exception:
            self.supply_counts[self.board.current_player] += 1
            raise

    def grasp_supply_stone(self, player: Player, x: float | None = None, y: float | None = None, z: float | None = None) -> None:
        if player not in {Player.BLACK, Player.WHITE}:
            raise ValueError("player must be BLACK or WHITE")
        if self.held_stone_player is not None:
            raise ValueError("robot is already holding a stone")
        if self.supply_counts[player] <= 0:
            raise ValueError(f"no {player.name.lower()} stones left in supply")
        self.supply_counts[player] -= 1
        if x is None or y is None or z is None:
            x, y, z = self.stone_supply_world(player)
        self.set_held_stone_world(x, y, z, player)

    def commit_held_stone_to_cell(self, row: int, col: int, update_robot_target: bool = True) -> dict[str, object]:
        if self.held_stone_player is None:
            raise ValueError("robot is not holding a stone")
        if self.held_stone_player != self.board.current_player:
            raise ValueError(
                f"held stone is {self.held_stone_player.name.lower()}, "
                f"but current player is {self.board.current_player.name.lower()}"
            )
        self.clear_held_stone()
        try:
            return self.step((row, col), update_robot_target=update_robot_target)
        except Exception:
            self.supply_counts[self.board.current_player] += 1
            raise

    @property
    def panda_joint_names(self) -> tuple[str, ...]:
        if self.robot_model != "panda":
            return ()
        return tuple(f"joint{idx}" for idx in range(1, 8)) + ("finger_joint1", "finger_joint2")

    @property
    def panda_arm_joint_names(self) -> tuple[str, ...]:
        if self.robot_model != "panda":
            return ()
        return tuple(f"joint{idx}" for idx in range(1, 8))

    @property
    def panda_site_names(self) -> tuple[str, str]:
        if self.robot_model != "panda":
            return ()
        return ("panda_ee_site", "panda_gripper_site")

    def panda_ee_world(self) -> tuple[float, float, float]:
        site_id = self._panda_site_id("panda_ee_site")
        mujoco.mj_forward(self.model, self.data)
        return tuple(float(value) for value in self.data.site_xpos[site_id])

    def panda_gripper_world(self) -> tuple[float, float, float]:
        site_id = self._panda_site_id("panda_gripper_site")
        mujoco.mj_forward(self.model, self.data)
        return tuple(float(value) for value in self.data.site_xpos[site_id])

    @property
    def so101_joint_names(self) -> tuple[str, ...]:
        if self.robot_model != "so101":
            return ()
        return self.so101_arm_joint_names + ("gripper",)

    @property
    def so101_arm_joint_names(self) -> tuple[str, ...]:
        if self.robot_model != "so101":
            return ()
        return ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

    @property
    def so101_site_names(self) -> tuple[str, str]:
        if self.robot_model != "so101":
            return ()
        return ("so101_ee_site", "so101_gripper_site")

    def so101_ee_world(self) -> tuple[float, float, float]:
        site_id = self._so101_site_id("so101_ee_site")
        mujoco.mj_forward(self.model, self.data)
        return tuple(float(value) for value in self.data.site_xpos[site_id])

    def so101_gripper_world(self) -> tuple[float, float, float]:
        site_id = self._so101_site_id("so101_gripper_site")
        mujoco.mj_forward(self.model, self.data)
        return tuple(float(value) for value in self.data.site_xpos[site_id])

    def so101_target_pose_for_cell(self, row: int, col: int, clearance: float = 0.075) -> tuple[float, float, float]:
        x, y, z = self.board_to_world(row, col)
        return x, y, z + clearance

    def panda_target_pose_for_cell(self, row: int, col: int, clearance: float = 0.180) -> tuple[float, float, float]:
        x, y, z = self.board_to_world(row, col)
        return x, y, z + clearance

    def solve_panda_ik(
        self,
        target_xyz: tuple[float, float, float],
        *,
        max_iterations: int = 250,
        tolerance: float = 0.002,
        damping: float = 1e-3,
        step_scale: float = 0.45,
    ) -> list[float]:
        self._require_panda()
        target = np.array(target_xyz, dtype=float)
        qpos_addrs = self._panda_arm_qpos_addrs()
        dof_addrs = self._panda_arm_dof_addrs()
        ranges = self._panda_arm_ranges()
        site_id = self._panda_site_id("panda_ee_site")
        original_qpos = self.data.qpos.copy()
        current = np.array([self.data.qpos[addr] for addr in qpos_addrs], dtype=float)
        seeds = [
            current,
            np.array([1.2, -0.5, 1.2, -2.6, 1.2, 1.8, -0.7853], dtype=float),
            np.array([-1.2, -0.5, -1.2, -2.6, -1.2, 1.8, -0.7853], dtype=float),
            np.array([0.8, -0.4, 0.8, -2.2, 0.8, 1.8, -0.7853], dtype=float),
            np.array([-0.8, -0.4, -0.8, -2.2, -0.8, 1.8, -0.7853], dtype=float),
            np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float),
        ]
        best_q = current
        best_error_norm = float("inf")

        for seed in seeds:
            q = np.clip(seed.copy(), ranges[:, 0], ranges[:, 1])
            for _ in range(max_iterations):
                for index, addr in enumerate(qpos_addrs):
                    self.data.qpos[addr] = q[index]
                mujoco.mj_forward(self.model, self.data)

                error = target - self.data.site_xpos[site_id]
                error_norm = float(np.linalg.norm(error))
                if error_norm < best_error_norm:
                    best_error_norm = error_norm
                    best_q = q.copy()
                if error_norm <= tolerance:
                    break

                jacp = np.zeros((3, self.model.nv), dtype=float)
                jacr = np.zeros((3, self.model.nv), dtype=float)
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
                jac = jacp[:, dof_addrs]
                system = jac @ jac.T + damping * np.eye(3)
                dq = jac.T @ np.linalg.solve(system, error)
                max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
                if max_abs > 0.10:
                    dq *= 0.10 / max_abs
                q = q + step_scale * dq
                q = np.clip(q, ranges[:, 0], ranges[:, 1])

        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)
        return [float(value) for value in best_q]

    def interpolate_panda_joint_trajectory(self, joint_targets: list[float], steps: int = 80) -> list[list[float]]:
        self._require_panda()
        if steps <= 0:
            raise ValueError("steps must be positive")
        if len(joint_targets) != 7:
            raise ValueError("joint_targets must contain 7 arm joint values")
        current = np.array([self.data.qpos[addr] for addr in self._panda_arm_qpos_addrs()], dtype=float)
        target = np.array(joint_targets, dtype=float)
        return [
            [float(value) for value in current + (target - current) * ((index + 1) / steps)]
            for index in range(steps)
        ]

    def set_panda_joint_targets(self, joint_targets: list[float], gripper: float = 1.0) -> None:
        self._require_panda()
        if len(joint_targets) != 7:
            raise ValueError("joint_targets must contain 7 arm joint values")
        for index, actuator_id in enumerate(self._panda_arm_actuator_ids()):
            self.data.ctrl[actuator_id] = joint_targets[index]
        self.set_panda_gripper(gripper)

    def set_panda_gripper(self, opening: float) -> None:
        self._require_panda()
        actuator_id = self._panda_gripper_actuator_id()
        opening = min(max(opening, 0.0), 1.0)
        self.data.ctrl[actuator_id] = opening * 255.0

    def move_panda_to_cell(
        self,
        row: int,
        col: int,
        *,
        clearance: float = 0.180,
        gripper: float = 1.0,
        trajectory_steps: int = 80,
        physics_steps_per_waypoint: int = 8,
    ) -> dict[str, object]:
        target_xyz = self.panda_target_pose_for_cell(row, col, clearance=clearance)
        joint_targets = self.solve_panda_ik(target_xyz)
        trajectory = self.interpolate_panda_joint_trajectory(joint_targets, steps=trajectory_steps)
        for waypoint in trajectory:
            self.set_panda_joint_targets(waypoint, gripper=gripper)
            self.simulate(physics_steps_per_waypoint)
        final_xyz = self.panda_ee_world()
        return {
            "target_cell": [row, col],
            "target_world_xyz": [float(value) for value in target_xyz],
            "joint_targets": joint_targets,
            "joint_trajectory": trajectory,
            "final_ee_world_xyz": [float(value) for value in final_xyz],
            "position_error_world": float(np.linalg.norm(np.array(target_xyz) - np.array(final_xyz))),
        }

    def solve_so101_ik(
        self,
        target_xyz: tuple[float, float, float],
        *,
        max_iterations: int = 300,
        tolerance: float = 0.003,
        damping: float = 1e-3,
        step_scale: float = 0.35,
        vertical_weight: float = 0.16,
    ) -> list[float]:
        self._require_so101()
        target = np.array(target_xyz, dtype=float)
        qpos_addrs = self._so101_arm_qpos_addrs()
        dof_addrs = self._so101_arm_dof_addrs()
        ranges = self._so101_arm_ranges()
        site_id = self._so101_site_id("so101_ee_site")
        original_qpos = self.data.qpos.copy()
        current = np.array([self.data.qpos[addr] for addr in qpos_addrs], dtype=float)
        seeds = [
            current,
            np.array([0.0, -0.65, 1.0, -0.35, 0.0], dtype=float),
            np.array([0.7, -0.65, 1.0, -0.35, 0.0], dtype=float),
            np.array([-0.7, -0.65, 1.0, -0.35, 0.0], dtype=float),
            np.array([0.0, 0.2, 1.2, -0.8, 0.0], dtype=float),
            np.array([0.9, 0.2, 1.1, -0.8, 0.0], dtype=float),
            np.array([-0.9, 0.2, 1.1, -0.8, 0.0], dtype=float),
        ]
        best_q = current
        best_score = float("inf")

        for seed in seeds:
            q = np.clip(seed.copy(), ranges[:, 0], ranges[:, 1])
            for _ in range(max_iterations):
                for index, addr in enumerate(qpos_addrs):
                    self.data.qpos[addr] = q[index]
                mujoco.mj_forward(self.model, self.data)

                error = target - self.data.site_xpos[site_id]
                error_norm = float(np.linalg.norm(error))
                site_xmat = np.array(self.data.site_xmat[site_id], dtype=float).reshape(3, 3)
                site_z = site_xmat[:, 2]
                desired_z = np.array([0.0, 0.0, 1.0], dtype=float)
                axis_error = desired_z - site_z
                axis_error_norm = float(np.linalg.norm(axis_error)) if vertical_weight > 0.0 else 0.0
                score = error_norm + vertical_weight * axis_error_norm
                if score < best_score:
                    best_score = score
                    best_q = q.copy()
                if error_norm <= tolerance and axis_error_norm <= 0.10:
                    break

                jacp = np.zeros((3, self.model.nv), dtype=float)
                jacr = np.zeros((3, self.model.nv), dtype=float)
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
                jac_pos = jacp[:, dof_addrs]
                if vertical_weight > 0.0:
                    jac_axis = -_skew(site_z) @ jacr[:, dof_addrs]
                    jac = np.vstack([jac_pos, vertical_weight * jac_axis])
                    stacked_error = np.concatenate([error, vertical_weight * axis_error])
                else:
                    jac = jac_pos
                    stacked_error = error
                system = jac.T @ jac + damping * np.eye(jac.shape[1])
                dq = np.linalg.solve(system, jac.T @ stacked_error)
                max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
                if max_abs > 0.08:
                    dq *= 0.08 / max_abs
                q = q + step_scale * dq
                q = np.clip(q, ranges[:, 0], ranges[:, 1])

        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)
        return [float(value) for value in best_q]

    def interpolate_so101_joint_trajectory(self, joint_targets: list[float], steps: int = 80) -> list[list[float]]:
        self._require_so101()
        if steps <= 0:
            raise ValueError("steps must be positive")
        if len(joint_targets) != 5:
            raise ValueError("joint_targets must contain 5 arm joint values")
        current = np.array([self.data.qpos[addr] for addr in self._so101_arm_qpos_addrs()], dtype=float)
        target = np.array(joint_targets, dtype=float)
        return [
            [float(value) for value in current + (target - current) * ((index + 1) / steps)]
            for index in range(steps)
        ]

    def set_so101_joint_targets(self, joint_targets: list[float], gripper: float = 1.0) -> None:
        self._require_so101()
        if len(joint_targets) != 5:
            raise ValueError("joint_targets must contain 5 arm joint values")
        qpos_addrs = self._so101_arm_qpos_addrs()
        for index, actuator_id in enumerate(self._so101_arm_actuator_ids()):
            self.data.ctrl[actuator_id] = joint_targets[index]
            self.data.qpos[qpos_addrs[index]] = joint_targets[index]
        self.set_so101_gripper(gripper)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_so101_gripper(self, opening: float) -> None:
        self._require_so101()
        actuator_id = self._so101_gripper_actuator_id()
        ctrlrange = self.model.actuator_ctrlrange[actuator_id]
        opening = min(max(opening, 0.0), 1.0)
        self.data.ctrl[actuator_id] = ctrlrange[0] + opening * (ctrlrange[1] - ctrlrange[0])
        joint_id = self._so101_joint_id("gripper")
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        joint_range = self.model.jnt_range[joint_id]
        self.data.qpos[qpos_addr] = float(joint_range[0] + opening * (joint_range[1] - joint_range[0]))
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def move_so101_to_cell(
        self,
        row: int,
        col: int,
        *,
        clearance: float = 0.075,
        gripper: float = 1.0,
        trajectory_steps: int = 80,
        physics_steps_per_waypoint: int = 8,
    ) -> dict[str, object]:
        target_xyz = self.so101_target_pose_for_cell(row, col, clearance=clearance)
        joint_targets = self.solve_so101_ik(target_xyz)
        trajectory = self.interpolate_so101_joint_trajectory(joint_targets, steps=trajectory_steps)
        for waypoint in trajectory:
            self.set_so101_joint_targets(waypoint, gripper=gripper)
            self.simulate(physics_steps_per_waypoint)
        final_xyz = self.so101_ee_world()
        return {
            "target_cell": [row, col],
            "target_world_xyz": [float(value) for value in target_xyz],
            "joint_targets": joint_targets,
            "joint_trajectory": trajectory,
            "final_ee_world_xyz": [float(value) for value in final_xyz],
            "position_error_world": float(np.linalg.norm(np.array(target_xyz) - np.array(final_xyz))),
        }

    def simulate(self, steps: int = 10) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def render(self, width: int = 900, height: int = 900, camera: str = "top") -> np.ndarray:
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            renderer.update_scene(self.data, camera=camera)
            return renderer.render()

    def export_model(self, path: str | Path) -> None:
        Path(path).write_text(self._model_xml, encoding="utf-8")

    def status_lines(self) -> tuple[str, str]:
        row, col = self.selected_cell
        left = f"cursor ({row}, {col})"
        if self.board.winner == Player.EMPTY:
            right = "draw"
        elif self.board.winner is not None:
            right = f"winner {self.board.winner.name.lower()}"
        else:
            right = f"turn {self.board.current_player.name.lower()}"
        return left, right

    def _collect_geom_ids(self) -> dict[str, int]:
        geom_ids: dict[str, int] = {}
        for row in range(self.board.size):
            for col in range(self.board.size):
                name = f"stone_{row}_{col}"
                geom_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        robot_geom_names = ("panda_hand", "panda_finger_left", "panda_finger_right") if self.robot_model == "kinematic" else ()
        for name in ("cursor", "held_stone", "active_stone", *robot_geom_names):
            geom_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return geom_ids

    def _reset_runtime_geoms(self) -> None:
        for row in range(self.board.size):
            for col in range(self.board.size):
                geom_id = self._geom_ids[f"stone_{row}_{col}"]
                self.model.geom_rgba[geom_id] = self._stone_rgba(Player.EMPTY)
        self.clear_held_stone()
        self.clear_active_stone()
        self._update_cursor_geom()
        if self.show_robot and self.robot_model == "kinematic":
            self._update_hand_geoms()

    def _update_cursor_geom(self) -> None:
        geom_id = self._geom_ids["cursor"]
        x, y, z = self.board_to_world(*self.selected_cell)
        self.model.geom_pos[geom_id] = np.array([x, y, z + 0.008], dtype=float)
        if self.robot_model in {"so101", "panda"}:
            rgba = np.array([0.05, 0.16, 0.20, 0.28], dtype=float)
        elif self.board.is_legal_move(*self.selected_cell):
            rgba = np.array([0.08, 0.72, 0.42, 0.72], dtype=float)
        else:
            rgba = np.array([0.84, 0.21, 0.17, 0.72], dtype=float)
        self.model.geom_rgba[geom_id] = rgba

    def _update_hand_geoms(self) -> None:
        if self.robot_model != "kinematic":
            return
        if self.robot_target_cell is None:
            x, y, _ = self._robot_home_world()
            z = 0.052
        else:
            x, y, z = self.board_to_world(*self.robot_target_cell)
            z += 0.034
        self._set_hand_geoms(x, y, z, gripper=1.0)

    def _set_hand_geoms(self, x: float, y: float, z: float, gripper: float) -> None:
        hand_id = self._geom_ids["panda_hand"]
        left_id = self._geom_ids["panda_finger_left"]
        right_id = self._geom_ids["panda_finger_right"]
        finger_offset = 0.007 + min(max(gripper, 0.0), 1.0) * 0.007
        self.model.geom_pos[hand_id] = np.array([x, y, z], dtype=float)
        self.model.geom_pos[left_id] = np.array([x - finger_offset, y, z - 0.014], dtype=float)
        self.model.geom_pos[right_id] = np.array([x + finger_offset, y, z - 0.014], dtype=float)

    def _require_panda(self) -> None:
        if not self.show_robot or self.robot_model != "panda":
            raise ValueError("Panda control is only available for robot_model='panda'")

    def _require_so101(self) -> None:
        if not self.show_robot or self.robot_model != "so101":
            raise ValueError("SO-101 control is only available for robot_model='so101'")

    def _panda_joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Panda joint is missing: {name}")
        return joint_id

    def _panda_site_id(self, name: str) -> int:
        self._require_panda()
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Panda site is missing: {name}")
        return site_id

    def _panda_actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Panda actuator is missing: {name}")
        return actuator_id

    def _panda_arm_qpos_addrs(self) -> list[int]:
        return [int(self.model.jnt_qposadr[self._panda_joint_id(name)]) for name in self.panda_arm_joint_names]

    def _panda_arm_dof_addrs(self) -> list[int]:
        return [int(self.model.jnt_dofadr[self._panda_joint_id(name)]) for name in self.panda_arm_joint_names]

    def _panda_arm_ranges(self) -> np.ndarray:
        return np.array([self.model.jnt_range[self._panda_joint_id(name)] for name in self.panda_arm_joint_names])

    def _panda_arm_actuator_ids(self) -> list[int]:
        return [self._panda_actuator_id(f"actuator{idx}") for idx in range(1, 8)]

    def _panda_gripper_actuator_id(self) -> int:
        return self._panda_actuator_id("actuator8")

    def _so101_joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"SO-101 joint is missing: {name}")
        return joint_id

    def _so101_site_id(self, name: str) -> int:
        self._require_so101()
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"SO-101 site is missing: {name}")
        return site_id

    def _so101_actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"SO-101 actuator is missing: {name}")
        return actuator_id

    def _active_stone_joint_id(self) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "active_stone_free")
        if joint_id < 0:
            raise ValueError("active stone free joint is missing")
        return joint_id

    def _so101_arm_qpos_addrs(self) -> list[int]:
        return [int(self.model.jnt_qposadr[self._so101_joint_id(name)]) for name in self.so101_arm_joint_names]

    def _so101_arm_dof_addrs(self) -> list[int]:
        return [int(self.model.jnt_dofadr[self._so101_joint_id(name)]) for name in self.so101_arm_joint_names]

    def _so101_arm_ranges(self) -> np.ndarray:
        return np.array([self.model.jnt_range[self._so101_joint_id(name)] for name in self.so101_arm_joint_names])

    def _so101_arm_actuator_ids(self) -> list[int]:
        return [self._so101_actuator_id(name) for name in self.so101_arm_joint_names]

    def _so101_gripper_actuator_id(self) -> int:
        return self._so101_actuator_id("gripper")

    def _robot_home_world(self) -> tuple[float, float, float]:
        board_half = self.board_extent / 2.0 + self.cell_size * 0.7
        return board_half + 0.030, -board_half - 0.030, 0.0

    def _set_stone_geom(self, row: int, col: int, player: Player) -> None:
        geom_id = self._geom_ids[f"stone_{row}_{col}"]
        self.model.geom_rgba[geom_id] = self._stone_rgba(player)

    def _stone_rgba(self, player: Player) -> np.ndarray:
        if player == Player.BLACK:
            return np.array([0.015, 0.014, 0.013, 1.0], dtype=float)
        if player == Player.WHITE:
            return np.array([0.92, 0.90, 0.85, 1.0], dtype=float)
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    def _stone_bowl_xml(self, player: Player) -> list[str]:
        x, y = self.stone_bowl_world(player)
        color_name = player.name.lower()
        stone_rgba = " ".join(f"{value:.3f}" for value in self._stone_rgba(player))
        reserve_offsets = [
            (0.000, 0.000),
            (-0.012, -0.006),
            (0.012, -0.005),
            (-0.007, 0.010),
            (0.009, 0.011),
        ]
        parts = [
            f"    <geom name=\"{color_name}_bowl_base\" type=\"cylinder\" pos=\"{x:.5f} {y:.5f} -0.00200\" "
            f"size=\"0.04300 0.00800\" material=\"bowl_dark\"/>",
            f"    <geom name=\"{color_name}_bowl_inner\" type=\"cylinder\" pos=\"{x:.5f} {y:.5f} 0.00600\" "
            f"size=\"0.03500 0.00400\" material=\"bowl_light\"/>",
            f"    <geom name=\"{color_name}_bowl_rim\" type=\"cylinder\" pos=\"{x:.5f} {y:.5f} 0.01000\" "
            f"size=\"0.04500 0.00300\" material=\"bowl_dark\"/>",
        ]
        for index, (dx, dy) in enumerate(reserve_offsets):
            parts.append(
                f"    <geom name=\"{color_name}_bowl_stone_{index}\" type=\"cylinder\" "
                f"pos=\"{x + dx:.5f} {y + dy:.5f} {0.014 + 0.0008 * index:.5f}\" "
                f"size=\"{self.stone_radius:.5f} {self.stone_height:.5f}\" rgba=\"{stone_rgba}\"/>"
            )
        return parts

    def _build_xml(self) -> str:
        board_half = self.board_extent / 2.0 + self.cell_size * 0.7
        board_thickness = 0.012
        grid_half = self.board_extent / 2.0
        line_width = 0.0011
        compiler_attrs = 'angle="radian"'
        if self.show_robot and self.robot_model == "panda":
            panda_asset_dir = self._panda_source_path().parent / "assets"
            compiler_attrs += f' meshdir="{panda_asset_dir}" autolimits="true"'
        elif self.show_robot and self.robot_model == "so101":
            so101_asset_dir = self._so101_source_path().parent / "assets"
            compiler_attrs += f' meshdir="{so101_asset_dir}" autolimits="true"'
        option_attrs = 'timestep="0.01" gravity="0 0 -9.81"'
        if self.show_robot and self.robot_model == "panda":
            option_attrs = 'timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"'
        elif self.show_robot and self.robot_model == "so101":
            option_attrs = 'timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="10" ls_iterations="20" impratio="10"'
        key_light_castshadow = "false" if self.show_robot and self.robot_model == "so101" else "true"
        xml_parts = [
            '<mujoco model="gomoku_board">',
            f"  <compiler {compiler_attrs}/>",
            f"  <option {option_attrs}/>",
            "  <visual>",
            "    <global offwidth=\"1400\" offheight=\"1000\"/>",
            "    <quality shadowsize=\"2048\"/>",
            "    <rgba haze=\"0.78 0.80 0.82 1\"/>",
            "    <headlight diffuse=\"0.55 0.55 0.50\" ambient=\"0.18 0.16 0.14\" specular=\"0.08 0.08 0.08\"/>",
            "  </visual>",
        ]
        if self.show_robot and self.robot_model == "panda":
            xml_parts.extend(["  <default>", *self._panda_children_xml("default", indent="    "), "  </default>"])
        elif self.show_robot and self.robot_model == "so101":
            xml_parts.extend(["  <default>", *self._so101_children_xml("default", indent="    "), "  </default>"])
        xml_parts.extend(
            [
            "  <asset>",
            "    <texture name=\"floor_tex\" type=\"2d\" builtin=\"checker\" width=\"512\" height=\"512\" rgb1=\"0.64 0.58 0.50\" rgb2=\"0.54 0.49 0.43\" mark=\"edge\" markrgb=\"0.47 0.42 0.36\"/>",
            "    <texture name=\"skybox_tex\" type=\"skybox\" builtin=\"flat\" width=\"32\" height=\"32\" rgb1=\"0.78 0.80 0.82\" rgb2=\"0.78 0.80 0.82\"/>",
            "    <material name=\"floor_mat\" texture=\"floor_tex\" texrepeat=\"4 4\" rgba=\"1 1 1 1\"/>",
            "    <material name=\"wall_mat\" rgba=\"0.79 0.76 0.69 1\"/>",
            "    <material name=\"table_mat\" rgba=\"0.42 0.28 0.17 1\" specular=\"0.18\" shininess=\"0.3\"/>",
            "    <material name=\"table_edge_mat\" rgba=\"0.22 0.14 0.09 1\"/>",
            "    <material name=\"board_mat\" rgba=\"0.91 0.67 0.34 1\" specular=\"0.12\" shininess=\"0.25\"/>",
            "    <material name=\"line_mat\" rgba=\"0.07 0.045 0.025 1\"/>",
            "    <material name=\"cup_mat\" rgba=\"0.12 0.32 0.40 1\" specular=\"0.2\" shininess=\"0.4\"/>",
            "    <material name=\"book_mat\" rgba=\"0.58 0.16 0.13 1\"/>",
            "    <material name=\"robot_white\" rgba=\"0.90 0.91 0.89 1\" specular=\"0.25\" shininess=\"0.55\"/>",
            "    <material name=\"robot_black\" rgba=\"0.08 0.085 0.09 1\" specular=\"0.18\" shininess=\"0.35\"/>",
            "    <material name=\"robot_accent\" rgba=\"0.04 0.42 0.62 1\" specular=\"0.25\" shininess=\"0.45\"/>",
            "    <material name=\"franka_label\" rgba=\"0.03 0.18 0.24 1\"/>",
            "    <material name=\"bowl_dark\" rgba=\"0.12 0.07 0.035 1\" specular=\"0.18\" shininess=\"0.25\"/>",
            "    <material name=\"bowl_light\" rgba=\"0.62 0.36 0.16 1\" specular=\"0.20\" shininess=\"0.22\"/>",
            ]
        )
        if self.show_robot and self.robot_model == "panda":
            xml_parts.extend(self._panda_children_xml("asset", indent="    "))
        elif self.show_robot and self.robot_model == "so101":
            xml_parts.extend(self._so101_children_xml("asset", indent="    "))
        xml_parts.extend(
            [
            "  </asset>",
            "  <worldbody>",
            f"    <light name=\"window_key\" pos=\"-0.9 -1.0 1.5\" dir=\"0.45 0.65 -1\" diffuse=\"0.95 0.88 0.76\" castshadow=\"{key_light_castshadow}\"/>",
            "    <light name=\"room_fill\" pos=\"0.8 0.7 1.1\" dir=\"-0.35 -0.2 -1\" diffuse=\"0.35 0.38 0.42\" castshadow=\"false\"/>",
            f"    <camera name=\"board_top\" pos=\"0 0 {board_half * 3.15:.5f}\" xyaxes=\"1 0 0 0 1 0\"/>",
            f"    <camera name=\"top\" pos=\"0 0 {board_half * 4.7:.5f}\" xyaxes=\"1 0 0 0 1 0\"/>",
            f"    <camera name=\"iso\" pos=\"{board_half * 2.8:.5f} {-board_half * 3.2:.5f} {board_half * 2.6:.5f}\" xyaxes=\"0.78 0.62 0 -0.36 0.45 0.82\"/>",
            f"    <camera name=\"robot_full\" pos=\"{board_half * 4.0:.5f} {-board_half * 4.1:.5f} {board_half * 2.8:.5f}\" xyaxes=\"0.72 0.69 0 -0.31 0.32 0.90\"/>",
            "    <geom name=\"floor\" type=\"plane\" pos=\"0 0 -0.065\" size=\"1.8 1.8 0.02\" material=\"floor_mat\"/>",
            "    <geom name=\"back_wall\" type=\"box\" pos=\"0 0.78 0.34\" size=\"1.5 0.018 0.42\" material=\"wall_mat\"/>",
            "    <geom name=\"side_wall\" type=\"box\" pos=\"-0.92 0.05 0.34\" size=\"0.018 1.2 0.42\" material=\"wall_mat\"/>",
            "    <geom name=\"table_top\" type=\"box\" pos=\"0 0 -0.035\" size=\"0.58 0.48 0.022\" material=\"table_mat\"/>",
            "    <geom name=\"table_front_edge\" type=\"box\" pos=\"0 -0.495 -0.023\" size=\"0.60 0.012 0.036\" material=\"table_edge_mat\"/>",
            "    <geom name=\"table_leg_1\" type=\"box\" pos=\"-0.51 -0.41 -0.17\" size=\"0.026 0.026 0.14\" material=\"table_edge_mat\"/>",
            "    <geom name=\"table_leg_2\" type=\"box\" pos=\"0.51 -0.41 -0.17\" size=\"0.026 0.026 0.14\" material=\"table_edge_mat\"/>",
            "    <geom name=\"table_leg_3\" type=\"box\" pos=\"-0.51 0.39 -0.17\" size=\"0.026 0.026 0.14\" material=\"table_edge_mat\"/>",
            "    <geom name=\"table_leg_4\" type=\"box\" pos=\"0.51 0.39 -0.17\" size=\"0.026 0.026 0.14\" material=\"table_edge_mat\"/>",
            "    <geom name=\"teacup\" type=\"cylinder\" pos=\"-0.39 -0.25 0.008\" size=\"0.035 0.030\" material=\"cup_mat\"/>",
            "    <geom name=\"cup_handle_top\" type=\"capsule\" fromto=\"-0.360 -0.250 0.032 -0.335 -0.250 0.026\" size=\"0.004\" material=\"cup_mat\"/>",
            "    <geom name=\"cup_handle_bottom\" type=\"capsule\" fromto=\"-0.360 -0.250 0.008 -0.335 -0.250 0.014\" size=\"0.004\" material=\"cup_mat\"/>",
            "    <geom name=\"notebook\" type=\"box\" pos=\"-0.37 0.23 -0.001\" size=\"0.075 0.105 0.006\" euler=\"0 0 0.17\" material=\"book_mat\"/>",
            f"    <geom name=\"board\" type=\"box\" pos=\"0 0 0\" size=\"{board_half:.5f} {board_half:.5f} {board_thickness:.5f}\" material=\"board_mat\"/>",
            ]
        )
        xml_parts.extend(self._stone_bowl_xml(Player.BLACK))
        xml_parts.extend(self._stone_bowl_xml(Player.WHITE))

        z = board_thickness + 0.0007
        for idx in range(self.board.size):
            offset = idx * self.cell_size - grid_half
            xml_parts.append(
                f"    <geom name=\"hline_{idx}\" type=\"box\" pos=\"0 {offset:.5f} {z:.5f}\" "
                f"size=\"{grid_half:.5f} {line_width:.5f} 0.0007\" material=\"line_mat\"/>"
            )
            xml_parts.append(
                f"    <geom name=\"vline_{idx}\" type=\"box\" pos=\"{offset:.5f} 0 {z:.5f}\" "
                f"size=\"{line_width:.5f} {grid_half:.5f} 0.0007\" material=\"line_mat\"/>"
            )

        for row in range(self.board.size):
            for col in range(self.board.size):
                x, y, stone_z = self.board_to_world(row, col)
                name = escape(f"stone_{row}_{col}")
                xml_parts.append(
                    f"    <geom name=\"{name}\" type=\"cylinder\" pos=\"{x:.5f} {y:.5f} {stone_z:.5f}\" "
                    f"size=\"{self.stone_radius:.5f} {self.stone_height:.5f}\" rgba=\"0 0 0 0\"/>"
                )

        xml_parts.append(
            f"    <geom name=\"cursor\" type=\"sphere\" pos=\"0 0 0.03\" size=\"0.00900\" rgba=\"0.08 0.72 0.42 0.72\"/>"
        )
        xml_parts.append(
            f"    <geom name=\"held_stone\" type=\"cylinder\" pos=\"0 0 0.03\" "
            f"size=\"{self.stone_radius:.5f} {self.stone_height:.5f}\" rgba=\"0 0 0 0\"/>"
        )
        xml_parts.append(
            f"    <body name=\"active_stone_body\" pos=\"0 0 -1\">"
            f"<freejoint name=\"active_stone_free\"/>"
            f"<geom name=\"active_stone\" type=\"cylinder\" "
            f"size=\"{self.stone_radius:.5f} {self.stone_height:.5f}\" rgba=\"0 0 0 0\" "
            f"mass=\"0.004\" friction=\"1.2 0.02 0.001\"/>"
            f"</body>"
        )

        top_level_parts: list[str] = []
        if self.show_robot:
            if self.robot_model == "panda":
                xml_parts.extend(self._panda_body_xml(board_half))
                top_level_parts.extend(self._panda_top_level_xml())
            elif self.robot_model == "so101":
                xml_parts.extend(self._so101_body_xml(board_half))
                top_level_parts.extend(self._so101_top_level_xml())
            else:
                xml_parts.extend(self._robot_xml(board_half))

        xml_parts.append("  </worldbody>")
        xml_parts.extend(top_level_parts)
        xml_parts.append("</mujoco>")
        return "\n".join(xml_parts)

    def _apply_robot_home_keyframe(self) -> None:
        if not self.show_robot:
            return
        if self.robot_model == "panda":
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id < 0:
                raise ValueError("Panda model is missing the 'home' keyframe")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            return
        if self.robot_model == "so101":
            self._apply_so101_home_pose()
            return

    def _apply_so101_home_pose(self) -> None:
        home_joint_targets = [1.1, -0.8, 1.2, -0.6, 0.0]
        home_gripper = 1.0
        qpos_addrs = self._so101_arm_qpos_addrs()
        for index, addr in enumerate(qpos_addrs):
            self.data.qpos[addr] = home_joint_targets[index]
        gripper_joint_id = self._so101_joint_id("gripper")
        gripper_qpos_addr = int(self.model.jnt_qposadr[gripper_joint_id])
        gripper_range = self.model.jnt_range[gripper_joint_id]
        self.data.qpos[gripper_qpos_addr] = float(gripper_range[0] + home_gripper * (gripper_range[1] - gripper_range[0]))
        self.data.qvel[:] = 0.0
        self.set_so101_joint_targets(home_joint_targets, gripper=home_gripper)
        mujoco.mj_forward(self.model, self.data)

    def _panda_source_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

    def _load_panda_tree(self) -> ET.Element:
        path = self._panda_source_path()
        if not path.exists():
            raise ValueError(f"Menagerie Panda MJCF is missing: {path}")
        return ET.parse(path).getroot()

    def _panda_body_xml(self, board_half: float) -> list[str]:
        root = self._load_panda_tree()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Menagerie Panda MJCF is missing worldbody")
        link0 = worldbody.find("body[@name='link0']")
        if link0 is None:
            raise ValueError("Menagerie Panda MJCF is missing body link0")
        base_x = board_half + 0.155
        base_y = 0.0
        link0 = deepcopy(link0)
        link0.set("pos", f"{base_x:.5f} {base_y:.5f} 0.00000")
        hand = link0.find(".//body[@name='hand']")
        if hand is None:
            raise ValueError("Menagerie Panda MJCF is missing body hand")
        hand.append(ET.Element("site", name="panda_ee_site", pos="0 0 0.1034", size="0.006", rgba="0.04 0.42 0.62 1"))
        hand.append(ET.Element("site", name="panda_gripper_site", pos="0 0 0.0700", size="0.004", rgba="0.08 0.72 0.42 1"))
        return ["    " + ET.tostring(link0, encoding="unicode")]

    def _panda_top_level_xml(self) -> list[str]:
        root = self._load_panda_tree()
        parts: list[str] = []
        for section_name in ("tendon", "equality", "actuator", "keyframe", "contact"):
            section = root.find(section_name)
            if section is not None:
                parts.append(f"  <{section_name}>")
                parts.extend(self._panda_children_xml(section_name, indent="    "))
                parts.append(f"  </{section_name}>")
        return parts

    def _panda_children_xml(self, tag: str, indent: str) -> list[str]:
        root = self._load_panda_tree()
        parent = root.find(tag)
        if parent is None:
            raise ValueError(f"Menagerie Panda MJCF is missing {tag}")
        return [indent + ET.tostring(deepcopy(child), encoding="unicode") for child in parent]

    def _so101_source_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"

    def _load_so101_tree(self) -> ET.Element:
        path = self._so101_source_path()
        if not path.exists():
            raise ValueError(f"Menagerie SO-101 MJCF is missing: {path}")
        return ET.parse(path).getroot()

    def _so101_body_xml(self, board_half: float) -> list[str]:
        root = self._load_so101_tree()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Menagerie SO-101 MJCF is missing worldbody")
        base = worldbody.find("body[@name='base']")
        if base is None:
            raise ValueError("Menagerie SO-101 MJCF is missing body base")
        base_x = self.board_extent / 2.0 + self.cell_size * 3.1
        base = deepcopy(base)
        base.set("pos", f"{base_x:.5f} 0.00000 0.01200")
        base.set("quat", "0 0 0 1")
        gripper = base.find(".//body[@name='gripper']")
        if gripper is None:
            raise ValueError("Menagerie SO-101 MJCF is missing body gripper")
        gripper.append(ET.Element("site", name="so101_ee_site", pos="0.012 -0.000218 -0.098127", size="0.004", rgba="0.04 0.42 0.62 1"))
        gripper.append(ET.Element("site", name="so101_gripper_site", pos="0.000 -0.000218 -0.084000", size="0.004", rgba="0.08 0.72 0.42 1"))
        return ["    " + ET.tostring(base, encoding="unicode")]

    def _so101_top_level_xml(self) -> list[str]:
        return ["  <actuator>", *self._so101_children_xml("actuator", indent="    "), "  </actuator>"]

    def _so101_children_xml(self, tag: str, indent: str) -> list[str]:
        root = self._load_so101_tree()
        parent = root.find(tag)
        if parent is None:
            raise ValueError(f"Menagerie SO-101 MJCF is missing {tag}")
        return [indent + ET.tostring(deepcopy(child), encoding="unicode") for child in parent]

    def _robot_xml(self, board_half: float) -> list[str]:
        target_x, target_y, _ = self._robot_home_world()
        base_x = board_half + 0.155
        base_y = -board_half + 0.115
        joints = [
            (base_x, base_y, 0.000),
            (base_x, base_y, 0.082),
            (base_x - 0.038, base_y + 0.060, 0.150),
            (base_x - 0.094, base_y + 0.126, 0.205),
            (target_x + 0.168, target_y - 0.124, 0.202),
            (target_x + 0.118, target_y - 0.085, 0.164),
            (target_x + 0.066, target_y - 0.046, 0.118),
            (target_x + 0.024, target_y - 0.018, 0.072),
        ]
        return [
            f"    <geom name=\"panda_base\" type=\"cylinder\" pos=\"{base_x:.5f} {base_y:.5f} 0.00000\" size=\"0.05500 0.03500\" material=\"robot_black\"/>",
            f"    <geom name=\"panda_pedestal\" type=\"cylinder\" pos=\"{base_x:.5f} {base_y:.5f} 0.04600\" size=\"0.04300 0.04600\" material=\"robot_white\"/>",
            self._capsule("panda_link1", joints[1], joints[2], 0.020, "robot_white"),
            self._sphere("panda_joint1", joints[1], 0.026, "robot_black"),
            self._capsule("panda_link2", joints[2], joints[3], 0.018, "robot_white"),
            self._sphere("panda_joint2", joints[2], 0.024, "robot_black"),
            self._capsule("panda_link3", joints[3], joints[4], 0.017, "robot_white"),
            self._sphere("panda_joint3", joints[3], 0.023, "robot_black"),
            self._capsule("panda_link4", joints[4], joints[5], 0.015, "robot_white"),
            self._sphere("panda_joint4", joints[4], 0.021, "robot_black"),
            self._capsule("panda_link5", joints[5], joints[6], 0.013, "robot_white"),
            self._sphere("panda_joint5", joints[5], 0.019, "robot_black"),
            self._capsule("panda_link6", joints[6], joints[7], 0.011, "robot_accent"),
            self._sphere("panda_joint6", joints[6], 0.017, "robot_black"),
            self._capsule("panda_link7", joints[7], (target_x, target_y, 0.056), 0.008, "robot_black"),
            self._sphere("panda_joint7", joints[7], 0.015, "robot_black"),
            f"    <geom name=\"franka_badge\" type=\"box\" pos=\"{base_x + 0.000:.5f} {base_y - 0.044:.5f} 0.07000\" size=\"0.02800 0.00200 0.00800\" material=\"franka_label\"/>",
            f"    <geom name=\"panda_hand\" type=\"box\" pos=\"{target_x:.5f} {target_y:.5f} 0.05200\" size=\"0.02300 0.01100 0.00700\" material=\"robot_black\"/>",
            f"    <geom name=\"panda_finger_left\" type=\"box\" pos=\"{target_x - 0.014:.5f} {target_y:.5f} 0.03800\" size=\"0.00450 0.01700 0.01200\" material=\"robot_black\"/>",
            f"    <geom name=\"panda_finger_right\" type=\"box\" pos=\"{target_x + 0.014:.5f} {target_y:.5f} 0.03800\" size=\"0.00450 0.01700 0.01200\" material=\"robot_black\"/>",
        ]

    def _capsule(
        self,
        name: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        radius: float,
        material: str,
    ) -> str:
        return (
            f"    <geom name=\"{name}\" type=\"capsule\" "
            f"fromto=\"{start[0]:.5f} {start[1]:.5f} {start[2]:.5f} {end[0]:.5f} {end[1]:.5f} {end[2]:.5f}\" "
            f"size=\"{radius:.5f}\" material=\"{material}\"/>"
        )

    def _sphere(self, name: str, pos: tuple[float, float, float], radius: float, material: str) -> str:
        return (
            f"    <geom name=\"{name}\" type=\"sphere\" pos=\"{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}\" "
            f"size=\"{radius:.5f}\" material=\"{material}\"/>"
        )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
