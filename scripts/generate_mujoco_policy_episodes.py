from __future__ import annotations

import argparse
import random
from pathlib import Path

from board import Player
from gomoku_ai.external_engine import build_piskvork_policy
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
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--engine-command",
        help="Piskvork/Gomocup engine command, for example a Rapfi executable path. Mutually exclusive with --checkpoint.",
    )
    parser.add_argument("--engine-timeout-turn-ms", type=int, default=1000)
    parser.add_argument("--engine-protocol-timeout-s", type=float, default=5.0)
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--assets-dir")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument(
        "--game-id-prefix",
        default=None,
        help="Unique game ID prefix for parallel production collection shards.",
    )
    parser.add_argument(
        "--episode-index-offset",
        type=int,
        default=0,
        help="Offset stored in observation episode indexes for parallel shards.",
    )
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
    parser.add_argument("--win-length", type=int, help="Override checkpoint win length.")
    parser.add_argument("--rule-set", choices=("free", "renju"))
    parser.add_argument("--center-opening", action="store_true")
    parser.add_argument("--no-center-opening", action="store_true")
    parser.add_argument("--max-moves", type=int)
    parser.add_argument(
        "--random-prefix-moves",
        type=int,
        default=0,
        help="Play this many random legal setup moves before recording teacher moves, for dataset diversity.",
    )
    parser.add_argument(
        "--random-prefix-seed",
        type=int,
        default=0,
        help="Seed for random legal setup prefixes.",
    )
    parser.add_argument(
        "--skip-failed-games",
        action="store_true",
        help="Continue collection when a teacher engine rejects a prefix or returns an unusable move.",
    )
    parser.add_argument("--policy-source", default="alphazero")
    parser.add_argument("--image-width", type=int, default=DEFAULT_TRAINING_IMAGE_SIZE)
    parser.add_argument("--image-height", type=int, default=DEFAULT_TRAINING_IMAGE_SIZE)
    parser.add_argument("--cameras", default=",".join(DEFAULT_TRAINING_CAMERAS))
    parser.add_argument("--robot-model", choices=("kinematic", "panda", "so101"), default="so101")
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.035,
        help="Physical spacing between board intersections in metres.",
    )
    parser.add_argument(
        "--stone-radius",
        type=float,
        default=0.012,
        help="Physical stone radius in metres; must be less than half the cell size.",
    )
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

    if bool(args.checkpoint) == bool(args.engine_command):
        parser.error("provide exactly one of --checkpoint or --engine-command")
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.board_size <= 0:
        parser.error("--board-size must be positive")
    if args.episode_index_offset < 0:
        parser.error("--episode-index-offset must be non-negative")
    if args.random_prefix_moves < 0:
        parser.error("--random-prefix-moves must be non-negative")
    if args.temperature < 0.0:
        parser.error("--temperature must be non-negative")
    if args.late_temperature < 0.0:
        parser.error("--late-temperature must be non-negative")
    if args.temperature_moves < 0:
        parser.error("--temperature-moves must be non-negative")
    if args.root_dirichlet_alpha <= 0.0:
        parser.error("--root-dirichlet-alpha must be positive")
    if args.cell_size <= 0.0:
        parser.error("--cell-size must be positive")
    if args.stone_radius <= 0.0 or args.stone_radius * 2.0 >= args.cell_size:
        parser.error("--stone-radius must be positive and less than half --cell-size")
    if not 0.0 <= args.root_exploration_fraction <= 1.0:
        parser.error("--root-exploration-fraction must be between 0 and 1")
    if args.center_opening and args.no_center_opening:
        parser.error("--center-opening and --no-center-opening cannot be used together")
    if args.engine_timeout_turn_ms <= 0:
        parser.error("--engine-timeout-turn-ms must be positive")
    if args.engine_protocol_timeout_s <= 0.0:
        parser.error("--engine-protocol-timeout-s must be positive")

    enforce_center_opening = None
    if args.center_opening:
        enforce_center_opening = True
    if args.no_center_opening:
        enforce_center_opening = False

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            parser.error(f"checkpoint not found: {checkpoint}")
        output_jsonl = Path(args.output_jsonl) if args.output_jsonl else default_mujoco_episode_output_path(checkpoint)
    else:
        checkpoint = None
        output_jsonl = Path(args.output_jsonl) if args.output_jsonl else Path("data/external_engine_mujoco_policy_episodes.jsonl")
    assets_dir = Path(args.assets_dir) if args.assets_dir else output_jsonl.parent / "assets"
    cameras = tuple(camera.strip() for camera in args.cameras.split(",") if camera.strip())
    if not cameras:
        parser.error("--cameras must contain at least one camera name")
    if (
        not args.allow_low_res_smoke
        and (args.image_width < MIN_TRAINING_IMAGE_SIZE or args.image_height < MIN_TRAINING_IMAGE_SIZE)
    ):
        parser.error(
            f"--image-width and --image-height must be at least {MIN_TRAINING_IMAGE_SIZE} for VLA collection; "
            "use --allow-low-res-smoke only for pipeline checks"
        )

    external_policy_config = None
    if checkpoint is not None:
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
        policy_source = args.policy_source
        checkpoint_label = str(checkpoint)
        game_id_prefix = args.game_id_prefix or f"{checkpoint.stem}-mujoco"
    else:
        rule_set = args.rule_set or "renju"
        center_opening = True if enforce_center_opening is None else enforce_center_opening
        external_policy_config = (
            args.engine_command,
            args.board_size,
            args.win_length or 5,
            rule_set,
            center_opening,
            args.engine_timeout_turn_ms,
            args.engine_protocol_timeout_s,
        )
        policy = _build_external_policy(external_policy_config)
        policy_source = args.policy_source if args.policy_source != "alphazero" else f"piskvork:{policy.name}"
        checkpoint_label = None
        game_id_prefix = args.game_id_prefix or f"{policy.name}-mujoco"
    total_records = 0
    failed_games = 0
    prefix_rng = random.Random(args.random_prefix_seed)
    try:
        for game_index in range(args.games):
            if external_policy_config is not None:
                close = getattr(policy, "close", None)
                if close is not None:
                    close()
                policy = _build_external_policy(external_policy_config)
            rule_set = args.rule_set or policy.rule_set
            center_opening = policy.enforce_center_opening if enforce_center_opening is None else enforce_center_opening
            env = GomokuMujocoEnv(
                board_size=policy.board_size,
                win_length=policy.win_length if args.win_length is None else args.win_length,
                rule_set=rule_set,
                enforce_center_opening=center_opening,
                cell_size=args.cell_size,
                stone_radius=args.stone_radius,
                show_robot=True,
                robot_model=args.robot_model,
            )
            prefix_count = _apply_random_prefix(env, args.random_prefix_moves, prefix_rng)
            collection_max_moves = None if args.max_moves is None else env.board.move_count + args.max_moves
            try:
                records = collect_mujoco_policy_episode(
                    env,
                    policy,
                    output_jsonl,
                    assets_dir,
                    game_id=f"{game_id_prefix}-{game_index + 1}",
                    episode_index=args.episode_index_offset + game_index,
                    policy_source=policy_source,
                    checkpoint=checkpoint_label,
                    max_moves=collection_max_moves,
                    cameras=cameras,
                    image_width=args.image_width,
                    image_height=args.image_height,
                    capture_phase_images=args.capture_phase_images,
                )
            except (RuntimeError, TimeoutError, ValueError) as exc:
                if not args.skip_failed_games:
                    raise
                failed_games += 1
                print(
                    f"skipped game {game_index + 1} after {prefix_count} prefix moves: {exc}",
                    flush=True,
                )
                close = getattr(policy, "close", None)
                if close is not None:
                    close()
                continue
            total_records += len(records)
    finally:
        close = getattr(policy, "close", None)
        if close is not None:
            close()

    print(
        f"wrote {total_records} MuJoCo move records from {args.games} games to {output_jsonl}; "
        f"failed_games={failed_games}",
        flush=True,
    )


def _build_external_policy(config: tuple[str, int, int, str, bool, int, float]):
    command, board_size, win_length, rule_set, center_opening, timeout_turn_ms, protocol_timeout_s = config
    return build_piskvork_policy(
        command,
        board_size=board_size,
        win_length=win_length,
        rule_set=rule_set,
        enforce_center_opening=center_opening,
        timeout_turn_ms=timeout_turn_ms,
        protocol_timeout_s=protocol_timeout_s,
    )


def _apply_random_prefix(env: GomokuMujocoEnv, moves: int, rng: random.Random) -> int:
    applied = 0
    while applied < moves and env.board.winner is None:
        legal_moves = env.board.legal_moves()
        if not legal_moves:
            break
        move = rng.choice(legal_moves)
        player = env.board.current_player
        env.step(move, update_robot_target=False)
        if player in {Player.BLACK, Player.WHITE}:
            env.supply_counts[player] = max(0, env.supply_counts[player] - 1)
        applied += 1
    return applied


if __name__ == "__main__":
    main()
