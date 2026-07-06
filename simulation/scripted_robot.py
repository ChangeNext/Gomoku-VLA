from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

from board import Player

from .mujoco_gomoku_env import GomokuMujocoEnv


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0

    def action(self, gripper: float) -> list[float]:
        return [self.x, self.y, self.z, self.qw, self.qx, self.qy, self.qz, gripper]


@dataclass(frozen=True)
class TrajectoryPoint:
    phase: str
    t: float
    pose: Pose
    gripper: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.pose.action(self.gripper)
        return data


def build_pick_place_action(
    env: GomokuMujocoEnv,
    target_cell: tuple[int, int],
    player: Player,
    *,
    lift_height: float = 0.075,
    approach_height: float = 0.055,
) -> dict[str, Any]:
    row, col = target_cell
    pick_xyz = env.stone_supply_world(player)
    place_xyz = env.board_to_world(row, col)
    home_xyz = env.robot_home_world()

    pick_pose = Pose(*pick_xyz)
    place_pose = Pose(*place_xyz)
    phases = [
        ("home", Pose(home_xyz[0], home_xyz[1], lift_height), 1.0),
        ("pre_pick", Pose(pick_pose.x, pick_pose.y, approach_height), 1.0),
        ("pick", pick_pose, 1.0),
        ("grasp", pick_pose, 0.0),
        ("lift", Pose(pick_pose.x, pick_pose.y, lift_height), 0.0),
        ("pre_place", Pose(place_pose.x, place_pose.y, approach_height), 0.0),
        ("place", place_pose, 0.0),
        ("release", place_pose, 1.0),
        ("retreat", Pose(place_pose.x, place_pose.y, lift_height), 1.0),
    ]
    trajectory = [
        TrajectoryPoint(phase=phase, t=index / max(1, len(phases) - 1), pose=pose, gripper=gripper)
        for index, (phase, pose, gripper) in enumerate(phases)
    ]
    placement_error_world = _distance(place_xyz, env.board_to_world(row, col))
    placement_error_cell = 0.0 if env.world_to_board(place_pose.x, place_pose.y) == target_cell else 1.0

    return {
        "controller_type": "scripted_kinematic_v1",
        "target_cell": [row, col],
        "player": player.name.lower(),
        "pick_pose_xyz": [pick_pose.x, pick_pose.y, pick_pose.z],
        "place_pose_xyz": [place_pose.x, place_pose.y, place_pose.z],
        "target_world_xyz": [place_pose.x, place_pose.y, place_pose.z],
        "ee_trajectory": [point.to_dict() for point in trajectory],
        "action": [point.pose.action(point.gripper) for point in trajectory],
        "action_names": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
        "joint_trajectory": None,
        "execution_success": True,
        "placement_error_world": placement_error_world,
        "placement_error_cell": placement_error_cell,
    }


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))
