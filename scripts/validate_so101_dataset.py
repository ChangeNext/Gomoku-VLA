from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


REQUIRED_INPUT_IMAGES = {"board_top_before", "wrist_cam_before"}
FORBIDDEN_INPUT_KEYS = {
    "selected_move",
    "action_index",
    "move_token",
    "target_cell",
    "target_world_xyz",
    "place_pose_xyz",
    "board_after",
    "board_after_flat",
    "policy_probs",
    "value",
    "action",
    "action_tokens",
    "joint_trajectory",
    "ee_trajectory",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return records


def validate_records(
    records: list[dict[str, Any]],
    *,
    root: Path,
    expected_games: int | None = None,
    expected_image_size: tuple[int, int] = (768, 768),
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_steps: set[tuple[str, int]] = set()
    board_hashes: Counter[str] = Counter()
    game_hashes: Counter[str] = Counter()
    opening_moves: Counter[str] = Counter()
    move_tokens: Counter[str] = Counter()
    results: Counter[str] = Counter()
    missing_images = 0
    corrupt_images = 0
    leakage_count = 0

    counters = Counter(
        legal=0,
        training_usable=0,
        safety=0,
        grasp=0,
        placement=0,
        execution=0,
        so101=0,
    )

    for index, record in enumerate(records):
        prefix = f"record[{index}]"
        game_id = str(record.get("game_id", ""))
        step = int(record.get("step", -1))
        games[game_id].append(record)
        key = (game_id, step)
        if key in seen_steps:
            errors.append(f"{prefix}: duplicate game/step {key}")
        seen_steps.add(key)

        legal = record.get("legal") is True
        counters["legal"] += int(legal)
        if not legal:
            errors.append(f"{prefix}: legal is not true")

        before = record.get("board_before")
        after = record.get("board_after")
        move = record.get("selected_move")
        player = record.get("current_player_value")
        if not _valid_board_transition(before, after, move, player):
            errors.append(f"{prefix}: board transition is not exactly the selected move")
        if isinstance(before, list):
            board_hashes[_stable_hash(before)] += 1

        action = record.get("robot_action") or {}
        controller = action.get("controller_type") == "so101_joint_trajectory_v1"
        counters["so101"] += int(controller)
        if not controller:
            errors.append(f"{prefix}: non-SO-101 controller {action.get('controller_type')!r}")

        safety = action.get("safety") or {}
        safety_ok = safety.get("ok") is True and all(
            (safety.get(name) or {}).get("ok") is True for name in ("pick", "place", "trace")
        )
        counters["safety"] += int(safety_ok)
        if not safety_ok:
            errors.append(f"{prefix}: safety validation failed")

        grasp = action.get("grasp_report") or {}
        grasp_ok = grasp.get("ok") is True
        counters["grasp"] += int(grasp_ok)
        if not grasp_ok:
            errors.append(f"{prefix}: grasp report failed: {grasp.get('reason')}")
        placement_ok = grasp.get("final_cell") == move and action.get("placement_error_cell") == 0.0
        counters["placement"] += int(placement_ok)
        if not placement_ok:
            errors.append(f"{prefix}: final cell or placement error does not match selected move")

        execution_ok = action.get("execution_success") is True
        counters["execution"] += int(execution_ok)
        if not execution_ok:
            errors.append(f"{prefix}: execution failed: {record.get('error')}")

        names = action.get("action_names")
        sequence = action.get("action")
        if not _valid_action_sequence(names, sequence):
            errors.append(f"{prefix}: invalid, empty, or non-finite joint action sequence")

        observation = record.get("observation") or {}
        metadata = observation.get("image_metadata") or {}
        usable = metadata.get("training_usable") is True
        counters["training_usable"] += int(usable)
        if not usable:
            errors.append(f"{prefix}: training_usable is not true")
        if metadata.get("selection_cursor_hidden") is not True:
            errors.append(f"{prefix}: selection cursor is not declared hidden")

        model_input = observation.get("model_input") or {}
        input_images = model_input.get("images") or {}
        if set(input_images) != REQUIRED_INPUT_IMAGES:
            errors.append(f"{prefix}: model input image keys are {sorted(input_images)}")
        leaked = sorted(_find_forbidden_keys(model_input))
        instruction = str(model_input.get("language_instruction", ""))
        if leaked or "<MOVE_" in instruction or _instruction_leaks_move(instruction, move):
            leakage_count += 1
            errors.append(f"{prefix}: input leakage detected keys={leaked}")

        image_paths = list((observation.get("images") or {}).values())
        for phase in observation.get("phase_images") or []:
            image_paths.extend((phase.get("images") or {}).values())
        qa_path = metadata.get("qa_contact_sheet")
        for value in image_paths:
            path = _resolve_path(str(value), root)
            if not path.exists():
                missing_images += 1
                errors.append(f"{prefix}: missing image {value}")
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    if image.size != expected_image_size:
                        errors.append(f"{prefix}: image {value} has size {image.size}")
            except (OSError, ValueError) as exc:
                corrupt_images += 1
                errors.append(f"{prefix}: corrupt image {value}: {exc}")
        if qa_path:
            path = _resolve_path(str(qa_path), root)
            if not path.exists():
                missing_images += 1
                errors.append(f"{prefix}: missing QA contact sheet {qa_path}")
            else:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, ValueError) as exc:
                    corrupt_images += 1
                    errors.append(f"{prefix}: corrupt QA contact sheet {qa_path}: {exc}")

        strategy = (observation.get("supervision") or {}).get("strategy") or {}
        target_sequence = (observation.get("supervision") or {}).get("target_sequence") or {}
        token = target_sequence.get("move_token")
        if token:
            move_tokens[str(token)] += 1
        if step == 0:
            opening_moves[str(move)] += 1

    for game_id, game_records in games.items():
        ordered = sorted(game_records, key=lambda item: int(item.get("step", -1)))
        steps = [int(record.get("step", -1)) for record in ordered]
        if steps != list(range(len(steps))):
            errors.append(f"game {game_id}: non-contiguous steps {steps}")
        natural_result = ordered[-1].get("winner") if ordered else None
        if not ordered or ordered[-1].get("terminal") is not True or natural_result not in {"black", "white", "draw"}:
            errors.append(f"game {game_id}: game is incomplete or ended without a game result")
        else:
            results[str(natural_result)] += 1
        game_hashes[_stable_hash([record.get("selected_move") for record in ordered])] += 1

    game_count = len(games)
    complete_games = sum(
        1
        for game in games.values()
        if game
        and sorted(game, key=lambda item: item["step"])[-1].get("terminal") is True
        and sorted(game, key=lambda item: item["step"])[-1].get("winner") in {"black", "white", "draw"}
    )
    if expected_games is not None and complete_games != expected_games:
        errors.append(f"expected {expected_games} complete games, found {complete_games}")
    if game_hashes and max(game_hashes.values()) > 1:
        warnings.append("one or more complete move sequences are duplicated")

    total = len(records)
    summary = {
        "status": "pass" if not errors else "fail",
        "total_records": total,
        "game_count": game_count,
        "complete_games": complete_games,
        "mean_moves_per_game": total / game_count if game_count else 0.0,
        "rates": {name: counters[name] / total if total else 0.0 for name in counters},
        "missing_images": missing_images,
        "corrupt_images": corrupt_images,
        "input_leakage_records": leakage_count,
        "duplicate_game_sequences": sum(count - 1 for count in game_hashes.values() if count > 1),
        "duplicate_board_states": sum(count - 1 for count in board_hashes.values() if count > 1),
        "results": dict(results),
        "opening_moves": dict(opening_moves),
        "move_tokens": dict(move_tokens),
        "errors": errors,
        "warnings": warnings,
    }
    return summary


def _valid_board_transition(before: Any, after: Any, move: Any, player: Any) -> bool:
    if not isinstance(before, list) or not isinstance(after, list) or not isinstance(move, list) or len(move) != 2:
        return False
    row, col = move
    try:
        if before[row][col] != 0 or after[row][col] != player:
            return False
        differences = sum(
            before[r][c] != after[r][c]
            for r in range(len(before))
            for c in range(len(before[r]))
        )
    except (IndexError, TypeError):
        return False
    return differences == 1


def _valid_action_sequence(names: Any, sequence: Any) -> bool:
    if not isinstance(names, list) or not names or not isinstance(sequence, list) or not sequence:
        return False
    return all(
        isinstance(point, list)
        and len(point) == len(names)
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
        for point in sequence
    )


def _find_forbidden_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_INPUT_KEYS or key.endswith("_after") or "phase_" in key:
                yield key
            yield from _find_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _find_forbidden_keys(child)


def _instruction_leaks_move(instruction: str, move: Any) -> bool:
    if not isinstance(move, list) or len(move) != 2:
        return False
    lowered = instruction.lower()
    row, col = move
    return f"row {row}" in lowered or f"column {col}" in lowered or f"col {col}" in lowered


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else root / path


def write_markdown(summary: dict[str, Any], path: Path, source: Path) -> None:
    rates = summary["rates"]
    lines = [
        "# SO-101 Dataset Quality Report",
        "",
        f"- Source: `{source}`",
        f"- Status: **{summary['status'].upper()}**",
        f"- Complete games: {summary['complete_games']} / {summary['game_count']}",
        f"- Move records: {summary['total_records']}",
        f"- Mean moves per game: {summary['mean_moves_per_game']:.2f}",
        f"- Legal rate: {rates['legal']:.4%}",
        f"- Training-usable rate: {rates['training_usable']:.4%}",
        f"- Safety pass rate: {rates['safety']:.4%}",
        f"- Grasp success rate: {rates['grasp']:.4%}",
        f"- Placement success rate: {rates['placement']:.4%}",
        f"- Execution success rate: {rates['execution']:.4%}",
        f"- SO-101 controller rate: {rates['so101']:.4%}",
        f"- Missing/corrupt images: {summary['missing_images']} / {summary['corrupt_images']}",
        f"- Input leakage records: {summary['input_leakage_records']}",
        f"- Duplicate game sequences: {summary['duplicate_game_sequences']}",
        "",
        "## Results",
        "",
        f"`{json.dumps(summary['results'], sort_keys=True)}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {error}" for error in summary["errors"][:200])
    if not summary["errors"]:
        lines.append("- None")
    if len(summary["errors"]) > 200:
        lines.append(f"- ... {len(summary['errors']) - 200} additional errors in JSON report")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    if not summary["warnings"]:
        lines.append("- None")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate production MuJoCo SO-101 Gomoku records.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--expected-games", type=int)
    parser.add_argument("--image-width", type=int, default=768)
    parser.add_argument("--image-height", type=int, default=768)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    source = Path(args.input_jsonl)
    summary = validate_records(
        read_jsonl(source),
        root=Path(args.root),
        expected_games=args.expected_games,
        expected_image_size=(args.image_width, args.image_height),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, Path(args.output_md), source)
    print(
        f"validated {summary['total_records']} records across {summary['game_count']} games: {summary['status']}",
        flush=True,
    )
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
