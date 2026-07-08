from __future__ import annotations

import argparse
from pathlib import Path

from gomoku_ai.inference import CheckpointPolicy
from simulation import GomokuMujocoEnv, collect_mujoco_policy_episode, default_mujoco_episode_output_path


MIN_TRAINING_IMAGE_SIZE = 224


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MuJoCo-rendered policy episodes with scripted pick/place action traces."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--assets-dir")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--rule-set", choices=("free", "renju"))
    parser.add_argument("--center-opening", action="store_true")
    parser.add_argument("--no-center-opening", action="store_true")
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--policy-source", default="alphazero")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=640)
    parser.add_argument("--cameras", default="top,iso,robot_full")
    parser.add_argument("--robot-model", choices=("kinematic", "panda", "so101"), default="so101")
    parser.add_argument(
        "--allow-low-res-smoke",
        action="store_true",
        help="Allow images smaller than 224px for quick pipeline smoke tests only.",
    )
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.center_opening and args.no_center_opening:
        parser.error("--center-opening and --no-center-opening cannot be used together")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        parser.error(f"checkpoint not found: {checkpoint}")

    enforce_center_opening = None
    if args.center_opening:
        enforce_center_opening = True
    if args.no_center_opening:
        enforce_center_opening = False

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else default_mujoco_episode_output_path(checkpoint)
    assets_dir = Path(args.assets_dir) if args.assets_dir else output_jsonl.parent / "assets"
    cameras = tuple(camera.strip() for camera in args.cameras.split(",") if camera.strip())
    if not cameras:
        parser.error("--cameras must contain at least one camera name")
    if (
        not args.allow_low_res_smoke
        and (args.image_width < MIN_TRAINING_IMAGE_SIZE or args.image_height < MIN_TRAINING_IMAGE_SIZE)
    ):
        parser.error(
            "--image-width and --image-height must be at least 224 for VLA collection; "
            "use --allow-low-res-smoke only for pipeline checks"
        )

    policy = CheckpointPolicy(checkpoint, device=args.device, simulations=args.simulations)
    total_records = 0
    for game_index in range(args.games):
        rule_set = args.rule_set or policy.rule_set
        center_opening = policy.enforce_center_opening if enforce_center_opening is None else enforce_center_opening
        env = GomokuMujocoEnv(
            board_size=policy.board_size,
            win_length=args.win_length,
            rule_set=rule_set,
            enforce_center_opening=center_opening,
            show_robot=True,
            robot_model=args.robot_model,
        )
        records = collect_mujoco_policy_episode(
            env,
            policy,
            output_jsonl,
            assets_dir,
            game_id=f"{checkpoint.stem}-mujoco-{game_index + 1}",
            episode_index=game_index,
            policy_source=args.policy_source,
            checkpoint=str(checkpoint),
            max_moves=args.max_moves,
            cameras=cameras,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        total_records += len(records)

    print(
        f"wrote {total_records} MuJoCo move records from {args.games} games to {output_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
