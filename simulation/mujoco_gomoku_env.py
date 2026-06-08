from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import escape

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

    def step(self, action: tuple[int, int]) -> dict[str, object]:
        row, col = action
        winner = self.board.place(row, col)
        self.selected_cell = (row, col)
        self.robot_target_cell = (row, col)
        self._set_stone_geom(row, col, Player(self.board.grid[row][col]))
        self._update_cursor_geom()
        self._update_hand_geoms()
        mujoco.mj_forward(self.model, self.data)
        return {
            "board": self.board.copy_state(),
            "current_player": self.board.current_player,
            "winner": winner,
            "done": winner is not None,
            "move_count": self.board.move_count,
        }

    def place_selected(self) -> dict[str, object]:
        return self.step(self.selected_cell)

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
        row, col = self.robot_target_cell or self.selected_cell
        x, y, _ = self.board_to_world(row, col)
        hand_id = self._geom_ids["panda_hand"]
        left_id = self._geom_ids["panda_finger_left"]
        right_id = self._geom_ids["panda_finger_right"]
        self.model.geom_pos[hand_id] = np.array([x, y, 0.052], dtype=float)
        self.model.geom_pos[left_id] = np.array([x - 0.012, y, 0.038], dtype=float)
        self.model.geom_pos[right_id] = np.array([x + 0.012, y, 0.038], dtype=float)

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
            "    <global offwidth=\"1200\" offheight=\"1200\"/>",
            "    <headlight diffuse=\"0.8 0.8 0.8\" ambient=\"0.2 0.2 0.2\" specular=\"0.1 0.1 0.1\"/>",
            "  </visual>",
            "  <asset>",
            "    <material name=\"board_mat\" rgba=\"0.86 0.64 0.36 1\"/>",
            "    <material name=\"line_mat\" rgba=\"0.08 0.06 0.04 1\"/>",
            "    <material name=\"robot_white\" rgba=\"0.88 0.90 0.90 1\" specular=\"0.25\" shininess=\"0.55\"/>",
            "    <material name=\"robot_black\" rgba=\"0.10 0.11 0.12 1\" specular=\"0.18\" shininess=\"0.35\"/>",
            "    <material name=\"robot_accent\" rgba=\"0.04 0.50 0.72 1\" specular=\"0.25\" shininess=\"0.45\"/>",
            "  </asset>",
            "  <worldbody>",
            "    <light name=\"key\" pos=\"0 -0.45 0.8\" dir=\"0 0 -1\" diffuse=\"0.9 0.9 0.85\" castshadow=\"false\"/>",
            f"    <camera name=\"top\" pos=\"0 0 {board_half * 4.7:.5f}\" xyaxes=\"1 0 0 0 1 0\"/>",
            f"    <camera name=\"iso\" pos=\"{board_half * 2.4:.5f} {-board_half * 2.8:.5f} {board_half * 2.3:.5f}\" xyaxes=\"0.78 0.62 0 -0.36 0.45 0.82\"/>",
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
        target_x, target_y, _ = self.board_to_world(*self.selected_cell)
        base_x = board_half + 0.105
        base_y = -board_half + 0.060
        joints = [
            (base_x, base_y, 0.026),
            (base_x, base_y, 0.092),
            (base_x - 0.026, base_y + 0.042, 0.142),
            (target_x + 0.182, target_y - 0.128, 0.184),
            (target_x + 0.128, target_y - 0.092, 0.154),
            (target_x + 0.078, target_y - 0.057, 0.125),
            (target_x + 0.035, target_y - 0.025, 0.088),
        ]
        return [
            f"    <geom name=\"panda_base\" type=\"cylinder\" pos=\"{joints[0][0]:.5f} {joints[0][1]:.5f} {joints[0][2]:.5f}\" size=\"0.04100 0.02600\" material=\"robot_black\"/>",
            self._capsule("panda_link1", joints[0], joints[1], 0.018, "robot_white"),
            self._sphere("panda_joint1", joints[1], 0.024, "robot_black"),
            self._capsule("panda_link2", joints[1], joints[2], 0.016, "robot_white"),
            self._sphere("panda_joint2", joints[2], 0.022, "robot_black"),
            self._capsule("panda_link3", joints[2], joints[3], 0.015, "robot_white"),
            self._sphere("panda_joint3", joints[3], 0.021, "robot_black"),
            self._capsule("panda_link4", joints[3], joints[4], 0.014, "robot_white"),
            self._sphere("panda_joint4", joints[4], 0.019, "robot_black"),
            self._capsule("panda_link5", joints[4], joints[5], 0.012, "robot_white"),
            self._sphere("panda_joint5", joints[5], 0.017, "robot_black"),
            self._capsule("panda_link6", joints[5], joints[6], 0.010, "robot_accent"),
            self._sphere("panda_joint6", joints[6], 0.015, "robot_black"),
            self._capsule("panda_link7", joints[6], (target_x, target_y, 0.053), 0.007, "robot_black"),
            f"    <geom name=\"panda_hand\" type=\"box\" pos=\"{target_x:.5f} {target_y:.5f} 0.05200\" size=\"0.02000 0.01000 0.00600\" material=\"robot_black\"/>",
            f"    <geom name=\"panda_finger_left\" type=\"box\" pos=\"{target_x - 0.012:.5f} {target_y:.5f} 0.03800\" size=\"0.00400 0.01500 0.01100\" material=\"robot_black\"/>",
            f"    <geom name=\"panda_finger_right\" type=\"box\" pos=\"{target_x + 0.012:.5f} {target_y:.5f} 0.03800\" size=\"0.00400 0.01500 0.01100\" material=\"robot_black\"/>",
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
