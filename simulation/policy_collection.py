from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from board import Player
from gomoku_ai.episode_recorder import EpisodeStepRecord, append_episode_record
from gomoku_ai.inference import MovePrediction
from robot_control import RobotSafetyController, SafetyReport

from .mujoco_gomoku_env import GomokuMujocoEnv
from .scripted_robot import build_pick_place_action

MIN_TRAINING_IMAGE_SIZE = 640
DEFAULT_TRAINING_IMAGE_SIZE = 768
DEFAULT_TRAINING_CAMERAS = ("board_top", "wrist_cam", "robot_full")
DEFAULT_MODEL_INPUT_CAMERAS = ("board_top", "wrist_cam")
DEFAULT_QA_CAMERAS = ("robot_full",)
SO101_GRIPPER_STONE_OFFSET_Z = -0.014
SO101_GRASP_DISTANCE_THRESHOLD = 0.055
SO101_PLACE_DISTANCE_THRESHOLD = 0.055


class MovePredictor(Protocol):
    def predict(self, board: object) -> MovePrediction:
        ...


def default_mujoco_episode_output_path(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint)
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return run_dir / "data" / f"{checkpoint_path.stem}_mujoco_policy_episodes.jsonl"


def collect_mujoco_policy_episode(
    env: GomokuMujocoEnv,
    predictor: MovePredictor,
    output_jsonl: str | Path,
    assets_dir: str | Path,
    *,
    game_id: str | None = None,
    episode_index: int = 0,
    policy_source: str = "alphazero",
    checkpoint: str | None = None,
    max_moves: int | None = None,
    cameras: tuple[str, ...] = DEFAULT_TRAINING_CAMERAS,
    image_width: int = DEFAULT_TRAINING_IMAGE_SIZE,
    image_height: int = DEFAULT_TRAINING_IMAGE_SIZE,
    capture_phase_images: bool = False,
) -> list[EpisodeStepRecord]:
    if env.board.winner is not None:
        raise ValueError("cannot collect an episode from a finished environment")
    max_steps = max_moves or env.board.size * env.board.size
    if max_steps <= 0:
        raise ValueError("max_moves must be positive")

    output_path = Path(output_jsonl)
    asset_root = Path(assets_dir)
    episode_id = game_id or str(uuid.uuid4())
    episode_asset_dir = asset_root / episode_id
    episode_asset_dir.mkdir(parents=True, exist_ok=True)
    records: list[EpisodeStepRecord] = []
    safety = RobotSafetyController(env)
    initial_move_count = env.board.move_count

    while env.board.winner is None and env.board.move_count < max_steps:
        step = len(records)
        board_ply_before = env.board.move_count
        board_before = env.board.copy_state()
        player = env.board.current_player
        supply_before = _supply_counts(env)
        held_before = _held_stone_name(env)
        prediction = predictor.predict(env.board)
        row, col = prediction.move
        legal = env.board.is_legal_move(row, col)
        env.set_selection(row, col, update_robot_target=False)
        images_before = _render_images(env, episode_asset_dir, step, "before", cameras, image_width, image_height)
        robot_action = build_pick_place_action(env, (row, col), player)
        pick_report = safety.validate_pick(player)
        place_report = safety.validate_place_cell(row, col, player)
        trace_report = safety.validate_action_trace(robot_action)
        safe_to_execute = pick_report.ok and place_report.ok and trace_report.ok
        error = None
        phase_images: list[dict[str, object]] = []
        if legal and safe_to_execute:
            if env.robot_model == "so101":
                robot_action, phase_images = _execute_so101_pick_place(
                    env,
                    episode_asset_dir,
                    step,
                    row,
                    col,
                    player,
                    cameras,
                    image_width,
                    image_height,
                    capture_phase_images=capture_phase_images,
                    base_action=robot_action,
                )
                if not robot_action["execution_success"]:
                    error = str(robot_action["grasp_report"]["reason"])
            else:
                phase_images = (
                    _render_phase_images(
                        env,
                        episode_asset_dir,
                        step,
                        robot_action,
                        player,
                        cameras,
                        image_width,
                        image_height,
                    )
                    if capture_phase_images
                    else []
                )
                pick_x, pick_y, pick_z = robot_action["pick_pose_xyz"]
                env.grasp_supply_stone(player, float(pick_x), float(pick_y), float(pick_z))
                env.commit_held_stone_to_cell(row, col, update_robot_target=True)
        else:
            error = _first_error(row, col, legal, pick_report, place_report, trace_report)
        images_after = _render_images(env, episode_asset_dir, step, "after", cameras, image_width, image_height)
        qa_contact_sheet = _write_contact_sheet(
            episode_asset_dir,
            step,
            images_before,
            images_after,
            phase_images,
        )
        supply_after = _supply_counts(env)
        held_after = _held_stone_name(env)
        robot_action["safety"] = {
            "ok": safe_to_execute,
            "pick": _report_dict(pick_report),
            "place": _report_dict(place_report),
            "trace": _report_dict(trace_report),
        }
        robot_action["supply_before"] = supply_before
        robot_action["supply_after"] = supply_after
        robot_action["held_stone_before"] = held_before
        robot_action["held_stone_after"] = held_after
        robot_action["robot_model"] = env.robot_model
        robot_action.setdefault("attachment_mode", "scripted_held_stone")
        robot_action["execution_success"] = bool(
            legal
            and safe_to_execute
            and env.board.is_on_board(row, col)
            and env.board.grid[row][col] == int(player.value)
            and robot_action["placement_error_cell"] == 0.0
            and robot_action.get("execution_success", True)
        )

        execution_failed = bool(legal and safe_to_execute and not robot_action.get("execution_success", False))
        will_stop = env.board.winner is not None or not legal or execution_failed or env.board.move_count >= max_steps
        observation = _build_observation(
            env,
            player,
            (row, col),
            robot_action,
            board_state=board_before,
            board_after=env.board.copy_state(),
            action_index=int(prediction.action_index),
            policy_probs=[float(value) for value in prediction.policy.tolist()],
            value=float(prediction.value),
            supply_before=supply_before,
            supply_after=supply_after,
            held_before=held_before,
            held_after=held_after,
            images_before=images_before,
            images_after=images_after,
            phase_images=phase_images,
            image_width=image_width,
            image_height=image_height,
            cameras=cameras,
            qa_contact_sheet=qa_contact_sheet,
            episode_index=episode_index,
            frame_index=step,
            prefix_moves=initial_move_count,
            board_ply_before=board_ply_before,
            board_ply_after=env.board.move_count,
            is_first=board_ply_before == 0,
            is_first_recorded_frame=step == 0,
            is_last=will_stop,
            is_terminal=env.board.winner is not None or not legal or not safe_to_execute or execution_failed,
        )
        record = EpisodeStepRecord(
            game_id=episode_id,
            step=step,
            timestamp=_now_iso(),
            board_before=board_before,
            board_after=env.board.copy_state(),
            board_size=env.board.size,
            win_length=env.board.win_length,
            rule_set=env.board.rule_set,
            enforce_center_opening=env.board.enforce_center_opening,
            current_player=player.name.lower(),
            current_player_value=int(player.value),
            selected_move=(row, col),
            action_index=int(prediction.action_index),
            policy_source=policy_source,
            policy_probs=[float(value) for value in prediction.policy.tolist()],
            value=float(prediction.value),
            legal=legal,
            used_tactical_move=prediction.used_tactical_move,
            winner=_winner_name(env.board.winner),
            winner_value=None if env.board.winner is None else int(env.board.winner.value),
            terminal=env.board.winner is not None or not legal or not safe_to_execute or execution_failed,
            checkpoint=checkpoint,
            robot_action=robot_action,
            observation=observation,
            error=error,
        )
        records.append(record)
        append_episode_record(output_path, record)
        if not legal or not safe_to_execute or execution_failed:
            break

    return records


def _execute_so101_pick_place(
    env: GomokuMujocoEnv,
    episode_asset_dir: Path,
    step: int,
    row: int,
    col: int,
    player: Player,
    cameras: tuple[str, ...],
    width: int,
    height: int,
    *,
    capture_phase_images: bool,
    base_action: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pick_x, pick_y, pick_z = env.stone_supply_world(player)
    place_x, place_y, place_z = env.board_to_world(row, col)
    home_xyz = env.so101_ee_world()
    phase_specs = [
        ("home", home_xyz, 1.0, 0),
        ("pre_pick", (pick_x, pick_y, pick_z + 0.090), 1.0, 14),
        ("pick", (pick_x, pick_y, pick_z - SO101_GRIPPER_STONE_OFFSET_Z), 1.0, 10),
        ("grasp", (pick_x, pick_y, pick_z - SO101_GRIPPER_STONE_OFFSET_Z), 0.0, 0),
        ("lift", (pick_x, pick_y, pick_z + 0.090), 0.0, 10),
        ("pre_place", (place_x, place_y, place_z + 0.090), 0.0, 16),
        ("place", (place_x, place_y, place_z - SO101_GRIPPER_STONE_OFFSET_Z), 0.0, 10),
        ("release", (place_x, place_y, place_z), 1.0, 0),
        ("retreat", home_xyz, 1.0, 14),
    ]
    joint_sequence: list[list[float]] = []
    ee_trajectory: list[dict[str, object]] = []
    phase_images: list[dict[str, object]] = []
    grasp_distance = float("inf")
    place_distance = float("inf")
    release_settle_error = float("inf")
    final_cell: list[int] | None = None
    reason: str | None = None

    def record_phase(phase_index: int, phase: str, xyz: tuple[float, float, float], gripper: float) -> None:
        action = _current_so101_action(env, gripper)
        joint_sequence.append(action)
        ee_trajectory.append(
            {
                "phase": phase,
                "t": phase_index / max(1, len(phase_specs) - 1),
                "pose": {"x": xyz[0], "y": xyz[1], "z": xyz[2], "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
                "gripper": gripper,
                "action": action,
            }
        )
        if capture_phase_images:
            images = _render_images(
                env,
                episode_asset_dir,
                step,
                f"phase_{phase_index:03d}_{_safe_phase_name(phase)}",
                cameras,
                width,
                height,
            )
            phase_images.append({"index": phase_index, "phase": phase, "t": phase_index / max(1, len(phase_specs) - 1), "images": images})

    for phase_index, (phase, xyz, gripper, steps) in enumerate(phase_specs):
        if steps > 0:
            joint_targets = env.solve_so101_ik(xyz)
            trajectory = env.interpolate_so101_joint_trajectory(joint_targets, steps=steps)
            for waypoint in trajectory:
                env.set_so101_joint_targets(waypoint, gripper=gripper)
                env.simulate(4)
                if env.active_stone_attached:
                    env.update_active_stone_at_so101_gripper(player, z_offset=SO101_GRIPPER_STONE_OFFSET_Z)
                joint_sequence.append(_current_so101_action(env, gripper))
        else:
            env.set_so101_gripper(gripper)
            env.simulate(12)

        if phase == "grasp":
            gripper_xyz = np.array(env.so101_gripper_world(), dtype=float)
            carried_xyz = gripper_xyz + np.array([0.0, 0.0, SO101_GRIPPER_STONE_OFFSET_Z], dtype=float)
            pick_xyz = np.array([pick_x, pick_y, pick_z], dtype=float)
            grasp_distance = float(np.linalg.norm(carried_xyz - pick_xyz))
            if grasp_distance <= SO101_GRASP_DISTANCE_THRESHOLD:
                env.attach_active_stone_to_so101_gripper(player, z_offset=SO101_GRIPPER_STONE_OFFSET_Z)
            else:
                reason = f"SO-101 gripper too far from supply stone: {grasp_distance:.4f}m"
        elif env.active_stone_attached:
            env.update_active_stone_at_so101_gripper(player, z_offset=SO101_GRIPPER_STONE_OFFSET_Z)

        if phase == "place":
            gripper_xyz = np.array(env.so101_gripper_world(), dtype=float)
            carried_xyz = gripper_xyz + np.array([0.0, 0.0, SO101_GRIPPER_STONE_OFFSET_Z], dtype=float)
            place_xyz = np.array([place_x, place_y, place_z], dtype=float)
            place_distance = float(np.linalg.norm(carried_xyz - place_xyz))
        if phase == "release" and env.active_stone_player == player:
            env.release_active_stone_at(place_x, place_y, place_z, player)
            env.simulate(24)
            active_world = env.active_stone_world()
            if active_world is not None:
                final_cell = list(env.world_to_board(active_world[0], active_world[1]))
                release_settle_error = float(np.linalg.norm(np.array(active_world) - np.array([place_x, place_y, place_z])))
            if reason is None and place_distance > SO101_PLACE_DISTANCE_THRESHOLD:
                reason = f"SO-101 gripper too far from target cell: {place_distance:.4f}m"
            if reason is None and final_cell != [row, col]:
                reason = f"active stone settled at {final_cell}, expected {[row, col]}"
            if reason is None:
                env.commit_active_stone_to_cell(row, col, update_robot_target=True)

        record_phase(phase_index, phase, xyz, gripper)

    execution_success = reason is None and env.board.grid[row][col] == int(player.value)
    placement_error_cell = 0.0 if execution_success else 1.0
    robot_action = dict(base_action)
    robot_action.update(
        {
            "controller_type": "so101_joint_trajectory_v1",
            "action_names": [*env.so101_arm_joint_names, "gripper"],
            "action": joint_sequence,
            "joint_trajectory": joint_sequence,
            "ee_trajectory": ee_trajectory,
            "attachment_mode": "constraint_style_active_stone",
            "grasp_report": {
                "mode": "constraint_style_active_stone",
                "physical_grasp": "constraint_backed_not_friction_only",
                "supply_container": f"{player.name.lower()}_bowl",
                "pick_source_world_xyz": [pick_x, pick_y, pick_z],
                "pick_source_outside_board": _outside_board_frame(env, pick_x, pick_y),
                "grasp_distance_world": grasp_distance,
                "place_distance_world": place_distance,
                "release_settle_error_world": release_settle_error,
                "final_cell": final_cell,
                "constraint_active_after_release": bool(env.active_stone_attached),
                "ok": execution_success,
                "reason": reason,
            },
            "execution_success": execution_success,
            "placement_error_cell": placement_error_cell,
            "placement_error_world": 0.0 if execution_success else release_settle_error,
        }
    )
    return robot_action, phase_images


def _outside_board_frame(env: GomokuMujocoEnv, x: float, y: float) -> bool:
    board_half = env.board_extent / 2.0 + env.cell_size * 0.7
    return abs(x) > board_half or abs(y) > board_half


def _current_so101_action(env: GomokuMujocoEnv, gripper: float) -> list[float]:
    qpos_addrs = [int(env.model.jnt_qposadr[env.model.joint(name).id]) for name in env.so101_arm_joint_names]
    return [float(env.data.qpos[addr]) for addr in qpos_addrs] + [float(gripper)]


def _render_images(
    env: GomokuMujocoEnv,
    episode_asset_dir: Path,
    step: int,
    timing: str,
    cameras: tuple[str, ...],
    width: int,
    height: int,
) -> dict[str, str]:
    images: dict[str, str] = {}
    cursor_id = env.model.geom("cursor").id
    saved_cursor_rgba = env.model.geom_rgba[cursor_id].copy()
    try:
        env.model.geom_rgba[cursor_id][3] = 0.0
        mujoco.mj_forward(env.model, env.data)
        for camera in cameras:
            image_path = episode_asset_dir / f"step_{step:03d}_{camera}_{timing}.png"
            Image.fromarray(env.render(width=width, height=height, camera=camera)).save(image_path)
            images[f"{camera}_{timing}"] = str(image_path)
    finally:
        env.model.geom_rgba[cursor_id] = saved_cursor_rgba
        mujoco.mj_forward(env.model, env.data)
    return images


def _render_phase_images(
    env: GomokuMujocoEnv,
    episode_asset_dir: Path,
    step: int,
    robot_action: dict[str, object],
    player: Player,
    cameras: tuple[str, ...],
    width: int,
    height: int,
) -> list[dict[str, object]]:
    held_id = env.model.geom("held_stone").id
    saved_held_player = env.held_stone_player
    saved_held_pos = env.model.geom_pos[held_id].copy()
    saved_held_rgba = env.model.geom_rgba[held_id].copy()
    saved_hand_geoms: dict[str, object] = {}
    if env.robot_model == "kinematic":
        for name in ("panda_hand", "panda_finger_left", "panda_finger_right"):
            geom_id = env.model.geom(name).id
            saved_hand_geoms[name] = env.model.geom_pos[geom_id].copy()

    phase_records: list[dict[str, object]] = []
    try:
        for index, point in enumerate(robot_action["ee_trajectory"]):
            phase = str(point["phase"])
            pose = point["pose"]
            x = float(pose["x"])
            y = float(pose["y"])
            z = float(pose["z"])
            gripper = float(point["gripper"])
            if env.robot_model == "kinematic":
                env.set_robot_hand_world(x, y, z, gripper=gripper)
            if _phase_carries_stone(phase):
                env.set_held_stone_world(x, y, z, player)
            else:
                env.clear_held_stone()
            images = _render_images(
                env,
                episode_asset_dir,
                step,
                f"phase_{index:03d}_{_safe_phase_name(phase)}",
                cameras,
                width,
                height,
            )
            phase_records.append(
                {
                    "index": index,
                    "phase": phase,
                    "t": float(point["t"]),
                    "images": images,
                }
            )
    finally:
        env.held_stone_player = saved_held_player
        env.model.geom_pos[held_id] = saved_held_pos
        env.model.geom_rgba[held_id] = saved_held_rgba
        for name, pos in saved_hand_geoms.items():
            env.model.geom_pos[env.model.geom(name).id] = pos
        mujoco.mj_forward(env.model, env.data)

    return phase_records


def _write_contact_sheet(
    episode_asset_dir: Path,
    step: int,
    images_before: dict[str, str],
    images_after: dict[str, str],
    phase_images: list[dict[str, object]],
) -> str | None:
    selected: list[tuple[str, str]] = []
    for key in ("board_top_before", "top_before", "wrist_cam_before", "robot_full_before", "iso_before"):
        if key in images_before and all(existing_key != key for existing_key, _ in selected):
            selected.append((key, images_before[key]))
        if len(selected) >= 3:
            break
    for preferred_phase in ("pick", "grasp", "place", "release"):
        for phase_record in phase_images:
            if phase_record.get("phase") != preferred_phase:
                continue
            images = phase_record.get("images")
            if isinstance(images, dict):
                key = next((name for name in images if name.startswith("robot_full_")), None)
                if key is not None:
                    selected.append((key, str(images[key])))
                    break
        if len(selected) >= 6:
            break
    for key in ("board_top_after", "top_after", "robot_full_after"):
        if key in images_after:
            selected.append((key, images_after[key]))
            break
    if not selected:
        return None

    tile_w = 320
    tile_h = 350
    cols = 2
    rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for index, (label, path) in enumerate(selected):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w - 20, tile_h - 48))
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((10, 10), label, fill=(0, 0, 0))
        tile.paste(image, ((tile_w - image.width) // 2, 40))
        sheet.paste(tile, ((index % cols) * tile_w, (index // cols) * tile_h))
    sheet_path = episode_asset_dir / f"step_{step:03d}_contact_sheet.png"
    sheet.save(sheet_path)
    if step == 0:
        sheet.save(episode_asset_dir / "contact_sheet.png")
    return str(sheet_path)


def _phase_carries_stone(phase: str) -> bool:
    return phase in {"grasp", "lift", "pre_place", "place", "release"}


def _safe_phase_name(phase: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in phase.lower())


def _phase_image_mode(robot_action: dict[str, object]) -> str:
    if robot_action.get("controller_type") == "so101_joint_trajectory_v1":
        return "so101_joint_trajectory_active_stone"
    return "scripted_visual_attachment"


def _filter_model_input_images(images_before: dict[str, str]) -> dict[str, str]:
    allowed_keys = {f"{camera}_before" for camera in DEFAULT_MODEL_INPUT_CAMERAS}
    filtered = {key: value for key, value in images_before.items() if key in allowed_keys}
    return filtered or dict(images_before)


def _build_observation(
    env: GomokuMujocoEnv,
    player: Player,
    target_cell: tuple[int, int],
    robot_action: dict[str, object],
    *,
    board_state: list[list[int]],
    board_after: list[list[int]],
    action_index: int,
    policy_probs: list[float],
    value: float,
    supply_before: dict[str, int],
    supply_after: dict[str, int],
    held_before: str | None,
    held_after: str | None,
    images_before: dict[str, str],
    images_after: dict[str, str],
    phase_images: list[dict[str, object]],
    image_width: int,
    image_height: int,
    cameras: tuple[str, ...],
    qa_contact_sheet: str | None,
    episode_index: int,
    frame_index: int,
    prefix_moves: int,
    board_ply_before: int,
    board_ply_after: int,
    is_first: bool,
    is_first_recorded_frame: bool,
    is_last: bool,
    is_terminal: bool,
) -> dict[str, object]:
    target_world = env.board_to_world(*target_cell)
    model_input_images = _filter_model_input_images(images_before)
    return {
        "language_instruction": f"play the strongest legal Gomoku move as {player.name.lower()}",
        "images": {**images_before, **images_after},
        "phase_images": phase_images,
        "image_metadata": {
            "width": image_width,
            "height": image_height,
            "cameras": list(cameras),
            "model_input_cameras": list(DEFAULT_MODEL_INPUT_CAMERAS),
            "qa_cameras": list(DEFAULT_QA_CAMERAS),
            "training_usable": image_width >= MIN_TRAINING_IMAGE_SIZE and image_height >= MIN_TRAINING_IMAGE_SIZE,
            "minimum_training_size": MIN_TRAINING_IMAGE_SIZE,
            "phase_images_enabled": bool(phase_images),
            "phase_image_mode": _phase_image_mode(robot_action),
            "qa_contact_sheet": qa_contact_sheet,
            "selection_cursor_hidden": True,
            "split": "train" if image_width >= MIN_TRAINING_IMAGE_SIZE and image_height >= MIN_TRAINING_IMAGE_SIZE else "smoke",
        },
        "model_input": {
            "language_instruction": f"play the strongest legal Gomoku move as {player.name.lower()}",
            "images": model_input_images,
            "state": {
                "board_flat": [value for row in board_state for value in row],
                "current_player_value": int(player.value),
                "robot_state": robot_action["action"][0] if robot_action.get("action") else None,
                "robot_model": env.robot_model,
                "supply_before": supply_before,
                "held_stone_before": held_before,
            },
        },
        "state": {
            "board_flat": [value for row in board_state for value in row],
            "current_player_value": int(player.value),
            "robot_state": robot_action["action"][0] if robot_action.get("action") else None,
            "robot_model": env.robot_model,
            "supply_before": supply_before,
            "held_stone_before": held_before,
            "safety": robot_action.get("safety"),
        },
        "supervision": {
            "target_sequence": _build_target_sequence(action_index, robot_action),
            "strategy": {
                "selected_move": [target_cell[0], target_cell[1]],
                "action_index": action_index,
                "policy_probs": policy_probs,
                "value": value,
                "board_after_flat": [value for row in board_after for value in row],
            },
            "execution": {
                "target_cell": [target_cell[0], target_cell[1]],
                "target_world_xyz": [target_world[0], target_world[1], target_world[2]],
                "supply_after": supply_after,
                "held_stone_after": held_after,
                "action": {
                    "format": robot_action["action_names"],
                    "sequence": robot_action["action"],
                    "controller_type": robot_action["controller_type"],
                    "attachment_mode": robot_action.get("attachment_mode"),
                    "grasp_report": robot_action.get("grasp_report"),
                },
            },
        },
        "episode_index": episode_index,
        "frame_index": frame_index,
        "prefix_moves": prefix_moves,
        "board_ply_before": board_ply_before,
        "board_ply_after": board_ply_after,
        "timestamp_s": frame_index,
        "is_first": is_first,
        "is_first_recorded_frame": is_first_recorded_frame,
        "is_last": is_last,
        "is_terminal": is_terminal,
    }


def _build_target_sequence(action_index: int, robot_action: dict[str, object]) -> dict[str, object]:
    move_token = _move_token(action_index)
    if robot_action.get("controller_type") == "so101_joint_trajectory_v1":
        action_tokens = [f"<ACT_SO101_{index:04d}>" for index, _ in enumerate(robot_action.get("action", []))]
        action_tokenization = "so101_joint_tokens_v1"
    else:
        action_tokens = [
            _action_phase_token(str(point["phase"]))
            for point in robot_action.get("ee_trajectory", [])
        ]
        action_tokenization = "scripted_phase_v1"
    return {
        "format": "autoregressive_move_then_action_v1",
        "tokens": [move_token, *action_tokens, "<EOS>"],
        "move_token": move_token,
        "action_tokens": action_tokens,
        "eos_token": "<EOS>",
        "move_tokenization": "board_cell_225_v1",
        "action_tokenization": action_tokenization,
        "continuous_action_source": "supervision.execution.action.sequence",
    }


def _move_token(action_index: int) -> str:
    if action_index < 0:
        raise ValueError("action_index must be non-negative")
    return f"<MOVE_{action_index:03d}>"


def _action_phase_token(phase: str) -> str:
    return f"<ACT_{_safe_phase_name(phase).upper()}>"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _winner_name(winner: Player | None) -> str | None:
    if winner is None:
        return None
    return winner.name.lower()


def _supply_counts(env: GomokuMujocoEnv) -> dict[str, int]:
    return {
        "black": int(env.supply_counts[Player.BLACK]),
        "white": int(env.supply_counts[Player.WHITE]),
    }


def _held_stone_name(env: GomokuMujocoEnv) -> str | None:
    return None if env.held_stone_player is None else env.held_stone_player.name.lower()


def _report_dict(report: SafetyReport) -> dict[str, object]:
    return {"ok": report.ok, "reason": report.reason}


def _first_error(
    row: int,
    col: int,
    legal: bool,
    pick_report: SafetyReport,
    place_report: SafetyReport,
    trace_report: SafetyReport,
) -> str:
    if not legal:
        return f"illegal move: row={row}, col={col}"
    for report in (pick_report, place_report, trace_report):
        if not report.ok:
            return report.reason or "unsafe robot action"
    return "unknown collection error"
