from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from board import Player
from simulation import GomokuMujocoEnv


@dataclass(frozen=True)
class WorkspaceLimits:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @classmethod
    def for_env(cls, env: GomokuMujocoEnv, *, margin: float = 0.12, max_z: float = 0.45) -> "WorkspaceLimits":
        half = env.board_extent / 2.0
        return cls(
            min_x=-half - margin,
            max_x=half + margin,
            min_y=-half - margin,
            max_y=half + margin,
            min_z=0.0,
            max_z=max_z,
        )

    def contains(self, xyz: Iterable[float]) -> bool:
        x, y, z = xyz
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )


@dataclass(frozen=True)
class SafetyReport:
    ok: bool
    reason: str | None = None

    def raise_if_unsafe(self) -> None:
        if not self.ok:
            raise ValueError(self.reason or "unsafe robot command")


class RobotSafetyController:
    def __init__(self, env: GomokuMujocoEnv, workspace: WorkspaceLimits | None = None) -> None:
        self.env = env
        self.workspace = workspace or WorkspaceLimits.for_env(env)

    def validate_pick(self, player: Player) -> SafetyReport:
        if player not in {Player.BLACK, Player.WHITE}:
            return SafetyReport(False, "player must be BLACK or WHITE")
        if self.env.held_stone_player is not None:
            return SafetyReport(False, "robot is already holding a stone")
        if self.env.supply_counts[player] <= 0:
            return SafetyReport(False, f"no {player.name.lower()} stones left in supply")
        pick_xyz = self.env.stone_supply_world(player)
        if not self.workspace.contains(pick_xyz):
            return SafetyReport(False, f"pick pose outside workspace: {pick_xyz}")
        return SafetyReport(True)

    def validate_place_cell(self, row: int, col: int, player: Player | None = None) -> SafetyReport:
        if not self.env.board.is_on_board(row, col):
            return SafetyReport(False, f"cell out of range: row={row}, col={col}")
        if not self.env.board.is_legal_move(row, col):
            return SafetyReport(False, f"illegal or occupied target cell: row={row}, col={col}")
        if player is not None and player != self.env.board.current_player:
            return SafetyReport(
                False,
                f"player mismatch: requested {player.name.lower()}, current {self.env.board.current_player.name.lower()}",
            )
        place_xyz = self.env.board_to_world(row, col)
        if not self.workspace.contains(place_xyz):
            return SafetyReport(False, f"place pose outside workspace: {place_xyz}")
        return SafetyReport(True)

    def validate_action_trace(self, robot_action: dict[str, object]) -> SafetyReport:
        for point in robot_action.get("ee_trajectory", []):
            pose = point.get("pose") if isinstance(point, dict) else None
            if not isinstance(pose, dict):
                return SafetyReport(False, "trajectory point is missing pose")
            xyz = (float(pose["x"]), float(pose["y"]), float(pose["z"]))
            if not self.workspace.contains(xyz):
                phase = point.get("phase", "unknown")
                return SafetyReport(False, f"trajectory phase {phase} outside workspace: {xyz}")
        return SafetyReport(True)
