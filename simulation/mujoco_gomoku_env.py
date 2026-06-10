from __future__ import annotations

import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape

if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from board import GomokuBoard, Player


class GomokuMujocoEnv:
    """MuJoCo-backed Gomoku scene with in-place updates for viewer use."""

    def __init__(
        self,
        board_size: int = 15,
        cell_size: float = 0.035,
        stone_radius: float = 0.012,
        stone_height: float = 0.006,
        show_robot: bool = True,
    ) -> None:
        self.board = GomokuBoard(size=board_size)
        self.cell_size = cell_size
        self.stone_radius = stone_radius
        self.stone_height = stone_height
        self.show_robot = show_robot
        self.selected_cell = (self.board.size // 2, self.board.size // 2)
        self.robot_target_cell: tuple[int, int] | None = None
        self._model_xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(self._model_xml)
        self.data = mujoco.MjData(self.model)
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
        for name in ("cursor", "panda_hand", "panda_finger_left", "panda_finger_right"):
            geom_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return geom_ids

    def _reset_runtime_geoms(self) -> None:
        for row in range(self.board.size):
            for col in range(self.board.size):
                geom_id = self._geom_ids[f"stone_{row}_{col}"]
                self.model.geom_rgba[geom_id] = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        self._update_cursor_geom()
        self._update_hand_geoms()

    def _update_cursor_geom(self) -> None:
        geom_id = self._geom_ids["cursor"]
        x, y, z = self.board_to_world(*self.selected_cell)
        self.model.geom_pos[geom_id] = np.array([x, y, z + 0.008], dtype=float)
        if self.board.is_legal_move(*self.selected_cell):
            rgba = np.array([0.08, 0.72, 0.42, 0.72], dtype=float)
        else:
            rgba = np.array([0.84, 0.21, 0.17, 0.72], dtype=float)
        self.model.geom_rgba[geom_id] = rgba

    def _update_hand_geoms(self) -> None:
        if self.robot_target_cell is None:
            x, y, _ = self._robot_home_world()
        else:
            x, y, _ = self.board_to_world(*self.robot_target_cell)
        hand_id = self._geom_ids["panda_hand"]
        left_id = self._geom_ids["panda_finger_left"]
        right_id = self._geom_ids["panda_finger_right"]
        self.model.geom_pos[hand_id] = np.array([x, y, 0.052], dtype=float)
        self.model.geom_pos[left_id] = np.array([x - 0.012, y, 0.038], dtype=float)
        self.model.geom_pos[right_id] = np.array([x + 0.012, y, 0.038], dtype=float)

    def _robot_home_world(self) -> tuple[float, float, float]:
        board_half = self.board_extent / 2.0 + self.cell_size * 0.7
        return board_half + 0.030, -board_half - 0.030, 0.0

    def _set_stone_geom(self, row: int, col: int, player: Player) -> None:
        geom_id = self._geom_ids[f"stone_{row}_{col}"]
        if player == Player.BLACK:
            rgba = np.array([0.015, 0.014, 0.013, 1.0], dtype=float)
        elif player == Player.WHITE:
            rgba = np.array([0.92, 0.90, 0.85, 1.0], dtype=float)
        else:
            rgba = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        self.model.geom_rgba[geom_id] = rgba

    def _build_xml(self) -> str:
        board_half = self.board_extent / 2.0 + self.cell_size * 0.7
        board_thickness = 0.012
        grid_half = self.board_extent / 2.0
        line_width = 0.0011
        xml_parts = [
            '<mujoco model="gomoku_board">',
            "  <compiler angle=\"radian\"/>",
            "  <option timestep=\"0.01\" gravity=\"0 0 -9.81\"/>",
            "  <visual>",
            "    <global offwidth=\"1400\" offheight=\"1000\"/>",
            "    <quality shadowsize=\"2048\"/>",
            "    <headlight diffuse=\"0.55 0.55 0.50\" ambient=\"0.18 0.16 0.14\" specular=\"0.08 0.08 0.08\"/>",
            "  </visual>",
            "  <asset>",
            "    <texture name=\"floor_tex\" type=\"2d\" builtin=\"checker\" width=\"512\" height=\"512\" rgb1=\"0.64 0.58 0.50\" rgb2=\"0.54 0.49 0.43\" mark=\"edge\" markrgb=\"0.47 0.42 0.36\"/>",
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
            "  </asset>",
            "  <worldbody>",
            "    <light name=\"window_key\" pos=\"-0.9 -1.0 1.5\" dir=\"0.45 0.65 -1\" diffuse=\"0.95 0.88 0.76\" castshadow=\"true\"/>",
            "    <light name=\"room_fill\" pos=\"0.8 0.7 1.1\" dir=\"-0.35 -0.2 -1\" diffuse=\"0.35 0.38 0.42\" castshadow=\"false\"/>",
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

        if self.show_robot:
            xml_parts.extend(self._robot_xml(board_half))

        xml_parts.extend(["  </worldbody>", "</mujoco>"])
        return "\n".join(xml_parts)

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
