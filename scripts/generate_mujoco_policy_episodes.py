from __future__ import annotations

import argparse
from pathlib import Path

from gomoku_ai.inference import CheckpointPolicy
from simulation import (
    DEFAULT_TRAINING_CAMERAS,
    DEFAULT_TRAINING_IMAGE_SIZE,
    MIN_TRAINING_IMAGE_SIZE,
    GomokuMujocoEnv,
    collect_mujoco_policy_episode,
    default_mujoco_episode_output_path,
)


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
    parser.add_argument(
        "--sample-moves",
        action="store_true",
        help="Sample teacher moves from the MCTS policy instead of always taking argmax.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="MCTS visit-count temperature for early moves. Keep 0.0 for deterministic argmax collection.",
    )
    parser.add_argument(
        "--temperature-moves",
        type=int,
        default=0,
        help="Use --temperature for this many opening moves, then switch to --late-temperature.",
    )
    parser.add_argument(
        "--late-temperature",
        type=float,
        default=0.0,
        help="MCTS temperature after --temperature-moves. Usually lower than --temperature.",
    )
    parser.add_argument(
        "--root-noise",
        action="store_true",
        help="Add root Dirichlet noise to MCTS for more diverse collection games.",
    )
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, help="Seed for sampled teacher move selection.")
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--rule-set", choices=("free", "renju"))
    parser.add_argument("--center-opening", action="store_true")
    parser.add_argument("--no-center-opening", action="store_true")
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--policy-source", default="alphazero")
    parser.add_argument("--image-width", type=int, default=DEFAULT_TRAINING_IMAGE_SIZE)
    parser.add_argument("--image-height", type=int, default=DEFAULT_TRAINING_IMAGE_SIZE)
    parser.add_argument("--cameras", default=",".join(DEFAULT_TRAINING_CAMERAS))
    parser.add_argument("--robot-model", choices=("kinematic", "panda", "so101"), default="so101")
    parser.add_argument(
        "--capture-phase-images",
        action="store_true",
        help="Also save one image set for each scripted pick/place phase.",
    )
    parser.add_argument(
        "--allow-low-res-smoke",
        action="store_true",
        help="Allow images smaller than 224px for quick pipeline smoke tests only.",
    )
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.temperature < 0.0:
        parser.error("--temperature must be non-negative")
    if args.late_temperature < 0.0:
        parser.error("--late-temperature must be non-negative")
    if args.temperature_moves < 0:
        parser.error("--temperature-moves must be non-negative")
    if args.root_dirichlet_alpha <= 0.0:
        parser.error("--root-dirichlet-alpha must be positive")
    if not 0.0 <= args.root_exploration_fraction <= 1.0:
        parser.error("--root-exploration-fraction must be between 0 and 1")
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

    policy = CheckpointPolicy(
        checkpoint,
        device=args.device,
        simulations=args.simulations,
        temperature=args.temperature,
        temperature_moves=args.temperature_moves,
        late_temperature=args.late_temperature,
        sample_moves=args.sample_moves,
        add_root_noise=args.root_noise,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        seed=args.seed,
    )
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
            capture_phase_images=args.capture_phase_images,
        )
        total_records += len(records)

    print(
        f"wrote {total_records} MuJoCo move records from {args.games} games to {output_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
