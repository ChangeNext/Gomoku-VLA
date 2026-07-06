from .mujoco_gomoku_env import GomokuMujocoEnv
from .scripted_robot import Pose, TrajectoryPoint, build_pick_place_action

__all__ = [
    "GomokuMujocoEnv",
    "Pose",
    "TrajectoryPoint",
    "build_pick_place_action",
    "collect_mujoco_policy_episode",
    "default_mujoco_episode_output_path",
]


def __getattr__(name: str):
    if name in {"collect_mujoco_policy_episode", "default_mujoco_episode_output_path"}:
        from .policy_collection import collect_mujoco_policy_episode, default_mujoco_episode_output_path

        exports = {
            "collect_mujoco_policy_episode": collect_mujoco_policy_episode,
            "default_mujoco_episode_output_path": default_mujoco_episode_output_path,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
