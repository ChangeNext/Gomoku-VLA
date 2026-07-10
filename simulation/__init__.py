from .mujoco_gomoku_env import GomokuMujocoEnv
from .scripted_robot import Pose, TrajectoryPoint, build_pick_place_action

__all__ = [
    "GomokuMujocoEnv",
    "Pose",
    "TrajectoryPoint",
    "build_pick_place_action",
    "collect_mujoco_policy_episode",
    "DEFAULT_MODEL_INPUT_CAMERAS",
    "DEFAULT_QA_CAMERAS",
    "DEFAULT_TRAINING_CAMERAS",
    "DEFAULT_TRAINING_IMAGE_SIZE",
    "default_mujoco_episode_output_path",
    "MIN_TRAINING_IMAGE_SIZE",
]


def __getattr__(name: str):
    if name in {
        "collect_mujoco_policy_episode",
        "DEFAULT_MODEL_INPUT_CAMERAS",
        "DEFAULT_QA_CAMERAS",
        "DEFAULT_TRAINING_CAMERAS",
        "DEFAULT_TRAINING_IMAGE_SIZE",
        "default_mujoco_episode_output_path",
        "MIN_TRAINING_IMAGE_SIZE",
    }:
        from .policy_collection import (
            DEFAULT_MODEL_INPUT_CAMERAS,
            DEFAULT_QA_CAMERAS,
            DEFAULT_TRAINING_CAMERAS,
            DEFAULT_TRAINING_IMAGE_SIZE,
            MIN_TRAINING_IMAGE_SIZE,
            collect_mujoco_policy_episode,
            default_mujoco_episode_output_path,
        )

        exports = {
            "collect_mujoco_policy_episode": collect_mujoco_policy_episode,
            "DEFAULT_MODEL_INPUT_CAMERAS": DEFAULT_MODEL_INPUT_CAMERAS,
            "DEFAULT_QA_CAMERAS": DEFAULT_QA_CAMERAS,
            "DEFAULT_TRAINING_CAMERAS": DEFAULT_TRAINING_CAMERAS,
            "DEFAULT_TRAINING_IMAGE_SIZE": DEFAULT_TRAINING_IMAGE_SIZE,
            "default_mujoco_episode_output_path": default_mujoco_episode_output_path,
            "MIN_TRAINING_IMAGE_SIZE": MIN_TRAINING_IMAGE_SIZE,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
