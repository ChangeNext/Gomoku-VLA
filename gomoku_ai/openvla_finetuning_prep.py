from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from board.gomoku import GomokuBoard, Player, RuleSet
from scripts.validate_so101_dataset import _find_forbidden_keys


DEFAULT_BOARD_SIZE = 15
DEFAULT_STAGE = "move_then_action_tokens"
VALID_STAGES = {"move_only", "teacher_move_then_action", "move_then_action_tokens"}
MOVE_TOKEN_RE = re.compile(r"^<MOVE_(\d{3})>$")
ACTION_TOKEN_RE = re.compile(r"^<ACT_SO101_\d+>$")
FORBIDDEN_INSTRUCTION_FRAGMENTS = ("row ", "column ", "col ", "target", "<MOVE_")


@dataclass(frozen=True)
class PreparationSummary:
    manifest_path: str
    output_dir: str
    samples: int
    stage: str
    move_tokens: int
    action_tokens: int
    artifacts: dict[str, str]
    warnings: list[str]


@dataclass(frozen=True)
class OpenVLAFinetuningSample:
    sample_id: str
    image_paths: dict[str, str]
    image_order: list[str]
    instruction: str
    target_text: str
    stage: str
    teacher_move_token: str | None = None


class OpenVLAFinetuningManifestAdapter:
    """Trainer-neutral dataset adapter for OpenVLA/OFT integration dry-runs."""

    def __init__(self, manifest_path: str | Path, *, stage: str = DEFAULT_STAGE, board_size: int = DEFAULT_BOARD_SIZE) -> None:
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}")
        self.manifest_path = Path(manifest_path)
        self.stage = stage
        self.board_size = board_size
        self.records = list(_read_jsonl(self.manifest_path))
        _validate_manifest_records(self.records, board_size=board_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OpenVLAFinetuningSample:
        record = self.records[index]
        return adapt_manifest_record(record, stage=self.stage)


def build_move_tokens(board_size: int = DEFAULT_BOARD_SIZE) -> list[str]:
    """Return one move token per board cell, using the repository's padded form."""

    if board_size <= 0:
        raise ValueError("board_size must be positive")
    width = len(str(board_size * board_size - 1))
    return [f"<MOVE_{index:0{width}d}>" for index in range(board_size * board_size)]


def move_token_to_action_index(token: str) -> int:
    match = MOVE_TOKEN_RE.match(token)
    if not match:
        raise ValueError(f"invalid move token {token!r}")
    return int(match.group(1))


def action_index_to_move_token(action_index: int, board_size: int = DEFAULT_BOARD_SIZE) -> str:
    if action_index < 0 or action_index >= board_size * board_size:
        raise ValueError(f"action_index {action_index} is outside board_size={board_size}")
    width = len(str(board_size * board_size - 1))
    return f"<MOVE_{action_index:0{width}d}>"


def adapt_manifest_record(record: dict[str, Any], *, stage: str) -> OpenVLAFinetuningSample:
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}")
    target = record["target"]
    tokens = [str(token) for token in target["tokens"]]
    if stage == "move_only":
        output_tokens = [tokens[0], "<EOS>"]
        teacher_move_token = None
    elif stage == "teacher_move_then_action":
        output_tokens = tokens[1:]
        teacher_move_token = tokens[0]
    else:
        output_tokens = tokens
        teacher_move_token = None
    model_input = record["input"]
    return OpenVLAFinetuningSample(
        sample_id=str(record.get("sample_id", "")),
        image_paths={str(key): str(value) for key, value in model_input["images"].items()},
        image_order=[str(key) for key in model_input["image_order"]],
        instruction=str(model_input["language_instruction"]),
        target_text=" ".join(output_tokens),
        stage=stage,
        teacher_move_token=teacher_move_token,
    )


def apply_special_tokens_to_tokenizer(tokenizer: Any, special_tokens: dict[str, Any]) -> int:
    """Add Gomoku fine-tuning tokens to a HuggingFace-like tokenizer.

    This is intentionally small and dependency-free so it can be imported on CPU
    machines without OpenVLA. GPU-side code should call model.resize_token_embeddings
    after this function returns.
    """

    tokens = list(special_tokens.get("move_tokens", [])) + list(special_tokens.get("action_tokens", []))
    eos_token = special_tokens.get("eos_token")
    if eos_token:
        tokens.append(str(eos_token))
    unique_tokens = list(dict.fromkeys(str(token) for token in tokens))
    if not unique_tokens:
        return 0
    return int(tokenizer.add_special_tokens({"additional_special_tokens": unique_tokens}))


def legal_move_token_mask(
    board_flat: Iterable[int],
    *,
    current_player_value: int,
    board_size: int = DEFAULT_BOARD_SIZE,
    win_length: int = 5,
    rule_set: RuleSet = "renju",
    enforce_center_opening: bool = False,
) -> list[bool]:
    """Return a boolean mask for legal `<MOVE_k>` logits using GomokuBoard."""

    values = [int(value) for value in board_flat]
    if len(values) != board_size * board_size:
        raise ValueError(f"board_flat must contain {board_size * board_size} values")
    board = GomokuBoard(
        size=board_size,
        win_length=win_length,
        rule_set=rule_set,
        enforce_center_opening=enforce_center_opening,
    )
    board.grid = [values[row * board_size : (row + 1) * board_size] for row in range(board_size)]
    board.move_count = sum(1 for value in values if value != Player.EMPTY.value)
    board.current_player = Player(current_player_value)
    legal = set(board.legal_moves())
    return [(index // board_size, index % board_size) in legal for index in range(board_size * board_size)]


def mask_move_logits(
    move_logits: Iterable[float],
    board_flat: Iterable[int],
    *,
    current_player_value: int,
    masked_value: float = float("-inf"),
    board_size: int = DEFAULT_BOARD_SIZE,
    win_length: int = 5,
    rule_set: RuleSet = "renju",
    enforce_center_opening: bool = False,
) -> list[float]:
    """Mask illegal move logits before selecting a generated move token."""

    logits = [float(value) for value in move_logits]
    if len(logits) != board_size * board_size:
        raise ValueError(f"move_logits must contain {board_size * board_size} values")
    mask = legal_move_token_mask(
        board_flat,
        current_player_value=current_player_value,
        board_size=board_size,
        win_length=win_length,
        rule_set=rule_set,
        enforce_center_opening=enforce_center_opening,
    )
    if not any(mask):
        raise ValueError("no legal moves are available")
    return [logit if is_legal else masked_value for logit, is_legal in zip(logits, mask)]


def prepare_openvla_finetuning_package(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    stage: str = DEFAULT_STAGE,
    board_size: int = DEFAULT_BOARD_SIZE,
    preview_samples: int = 5,
) -> PreparationSummary:
    """Create OpenVLA fine-tuning preparation files without running training.

    The generated files are intentionally trainer-neutral. They document token
    resizing, trainable new embeddings/lm_head rows, discrete action-token CE,
    and legality masking requirements for the later GPU-side implementation.
    """

    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}")
    if preview_samples < 0:
        raise ValueError("preview_samples must be non-negative")

    manifest = Path(manifest_path)
    records = list(_read_jsonl(manifest))
    _validate_manifest_records(records, board_size=board_size)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    move_tokens = build_move_tokens(board_size)
    action_tokens = sorted(_collect_action_tokens(records))
    special_tokens = {
        "move_tokens": move_tokens,
        "action_tokens": action_tokens,
        "eos_token": "<EOS>",
        "notes": [
            "Add these as additional special tokens before resizing token embeddings.",
            "Do not freeze newly added embedding rows or matching lm_head rows.",
            "Keep move tokens separate from OpenVLA action-bin tokens to avoid semantic collisions.",
        ],
    }
    run_config = _build_run_config(
        base_model=base_model,
        manifest_path=manifest,
        stage=stage,
        board_size=board_size,
        move_tokens=move_tokens,
        action_tokens=action_tokens,
    )
    prompt = _build_training_prompt(run_config)
    readme = _build_readme(run_config)

    artifacts = {
        "special_tokens": "special_tokens.json",
        "run_config": "run_config.json",
        "training_prompt": "training_prompt.md",
        "readme": "README.md",
        "preview": "dataset_preview.jsonl",
    }
    _write_json(target_dir / artifacts["special_tokens"], special_tokens)
    _write_json(target_dir / artifacts["run_config"], run_config)
    (target_dir / artifacts["training_prompt"]).write_text(prompt, encoding="utf-8")
    (target_dir / artifacts["readme"]).write_text(readme, encoding="utf-8")
    _write_preview(target_dir / artifacts["preview"], records[:preview_samples], stage=stage)

    warnings = _build_warnings(records, action_tokens=action_tokens)
    return PreparationSummary(
        manifest_path=str(manifest),
        output_dir=str(target_dir),
        samples=len(records),
        stage=stage,
        move_tokens=len(move_tokens),
        action_tokens=len(action_tokens),
        artifacts={key: str(target_dir / value) for key, value in artifacts.items()},
        warnings=warnings,
    )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _validate_manifest_records(records: list[dict[str, Any]], *, board_size: int) -> None:
    if not records:
        raise ValueError("manifest contains no records")
    expected_move_tokens = set(build_move_tokens(board_size))
    errors: list[str] = []
    for index, record in enumerate(records):
        prefix = f"sample[{index}]"
        model_input = record.get("input") or {}
        leaked = sorted(_find_forbidden_keys(model_input))
        if leaked:
            errors.append(f"{prefix}: target leakage in input keys {leaked}")
        instruction = str(model_input.get("language_instruction", ""))
        lowered = instruction.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_INSTRUCTION_FRAGMENTS):
            errors.append(f"{prefix}: target-like instruction {instruction!r}")

        target = record.get("target") or {}
        tokens = [str(token) for token in target.get("tokens") or []]
        move_token = str(target.get("move_token", ""))
        action_index = target.get("action_index")
        if not tokens or tokens[0] != move_token or tokens[-1] != "<EOS>":
            errors.append(f"{prefix}: target tokens must be <MOVE_k> ... <EOS>")
        if move_token not in expected_move_tokens:
            errors.append(f"{prefix}: move token {move_token!r} is outside {board_size}x{board_size}")
        if isinstance(action_index, int):
            expected = action_index_to_move_token(action_index, board_size)
            if move_token != expected:
                errors.append(f"{prefix}: move token {move_token!r} does not match action_index {action_index}")
        else:
            errors.append(f"{prefix}: missing integer action_index")
        for token in tokens[1:-1]:
            if not ACTION_TOKEN_RE.match(token):
                errors.append(f"{prefix}: non-discrete SO-101 action token {token!r}")
                break
    if errors:
        raise ValueError("\n".join(errors[:50]))


def _collect_action_tokens(records: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for record in records:
        for token in (record.get("target") or {}).get("tokens") or []:
            value = str(token)
            if ACTION_TOKEN_RE.match(value):
                tokens.add(value)
    return tokens


def _build_run_config(
    *,
    base_model: str,
    manifest_path: Path,
    stage: str,
    board_size: int,
    move_tokens: list[str],
    action_tokens: list[str],
) -> dict[str, Any]:
    return {
        "base_model": base_model,
        "manifest_path": str(manifest_path),
        "stage": stage,
        "board_size": board_size,
        "model_inputs": {
            "images": ["board_top_before", "wrist_cam_before"],
            "language_instruction": "input.language_instruction",
            "forbidden_input_values": [
                "row/column target text",
                "target_cell",
                "target_world_xyz",
                "selected_move",
                "action_index",
                "policy_probs",
                "value",
            ],
        },
        "targets": {
            "move_token": "target.move_token",
            "autoregressive_text": "target.text",
            "action_tokens": "target.tokens[1:-1]",
        },
        "tokenizer": {
            "add_special_tokens_from": "special_tokens.json",
            "new_move_token_count": len(move_tokens),
            "new_action_token_count": len(action_tokens),
            "resize_embeddings": True,
            "train_new_embedding_rows": True,
            "train_new_lm_head_rows": True,
            "do_not_reuse_openvla_action_bins_for_move_tokens": True,
        },
        "loss": {
            "move_token_loss": "cross_entropy",
            "action_loss": "cross_entropy_over_discrete_so101_action_tokens",
            "recommended_start_lambdas": {
                "move": 1.0,
                "action": 1.0 if stage != "move_only" else 0.0,
            },
            "reason": "CE for both move and discrete action tokens avoids CE/MSE scale mismatch.",
        },
        "inference_safety": {
            "legality_mask_required": True,
            "legality_source": "board.gomoku.GomokuBoard",
            "mask_before_move_selection": [
                "occupied_cells",
                "rule_set_illegal_moves",
                "renju_forbidden_moves_when_applicable",
            ],
            "execution_gate": "robot_control.RobotSafetyController",
        },
        "data_strategy": {
            "phase_1": "large simulated board image pretraining for board_top_before -> <MOVE_k>",
            "phase_2": "teacher <MOVE_k> plus wrist_cam_before -> SO-101 action tokens",
            "phase_3": "full board_top_before,wrist_cam_before -> <MOVE_k> <ACT_...> <EOS>",
            "sim_to_real_note": "Use real robot data mainly to adapt execution/action observations after image pretraining.",
        },
    }


def _build_training_prompt(run_config: dict[str, Any]) -> str:
    return f"""# GPU Training Prompt: Gomoku OpenVLA Fine-tuning

You are preparing OpenVLA/OFT fine-tuning for Gomoku-VLA.

Use this run config:

```json
{json.dumps(run_config, indent=2, sort_keys=True)}
```

Implementation requirements:

1. Load the exported manifest from `manifest_path`.
2. Add all tokens from `special_tokens.json` as additional special tokens.
3. Call the model embedding resize API after tokenizer expansion.
4. Keep newly added embedding rows and matching `lm_head` rows trainable, even if the base VLM/VLA is otherwise LoRA-tuned or partially frozen.
5. Train move and SO-101 action tokens with autoregressive cross-entropy. Do not mix continuous MSE action loss in the first GPU run.
6. Use `board_top_before` and `wrist_cam_before` as model images. Keep `robot_full_before` as QA/ablation only.
7. Do not feed `selected_move`, `action_index`, `target_cell`, `target_world_xyz`, `policy_probs`, or `value` as model inputs.
8. During inference, parse the board state and mask illegal `<MOVE_k>` logits with `board.gomoku.GomokuBoard` before selecting a move.
9. Before robot execution, pass the selected legal move through `robot_control.RobotSafetyController`.

Recommended first run:

- Stage: `move_only`
- Objective: verify image/instruction -> first generated token `<MOVE_k>`
- Stop before action execution if move-token accuracy or legality rate is poor.

Then run:

- Stage: `teacher_move_then_action`
- Objective: action-token generation conditioned on a known legal move token.

Finally run:

- Stage: `move_then_action_tokens`
- Objective: full sequence `<MOVE_k> <ACT_SO101_...> <EOS>`.
"""


def _build_readme(run_config: dict[str, Any]) -> str:
    return f"""# Gomoku OpenVLA Fine-tuning Preparation

This directory is a GPU-side preparation package. It does not run training by
itself; it records the tokenizer, loss, data, and safety requirements needed to
connect the exported Gomoku manifest to an OpenVLA/OFT trainer.

## Inputs

- Manifest: `{run_config['manifest_path']}`
- Images: `board_top_before`, `wrist_cam_before`
- Instruction: non-leaking natural-language move request

## Targets

- First token: `<MOVE_k>` for the selected board cell
- Remaining tokens: discrete SO-101 action tokens
- End token: `<EOS>`

## Critical Rules

- Resize tokenizer/model embeddings after adding tokens.
- Train new embedding and `lm_head` rows.
- Use CE for move and action tokens in the first implementation.
- Mask illegal moves at inference with `GomokuBoard`.
- Keep robot safety checks outside the learned model.
"""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_preview(path: Path, records: list[dict[str, Any]], *, stage: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            sample = adapt_manifest_record(record, stage=stage)
            preview = {
                "sample_id": sample.sample_id,
                "input": {
                    "language_instruction": sample.instruction,
                    "image_order": sample.image_order,
                    "images": sample.image_paths,
                },
                "target_text": sample.target_text,
            }
            if sample.teacher_move_token is not None:
                preview["input"]["teacher_move_token"] = sample.teacher_move_token
            handle.write(json.dumps(preview, sort_keys=True) + "\n")


def _build_warnings(records: list[dict[str, Any]], *, action_tokens: list[str]) -> list[str]:
    warnings: list[str] = []
    move_counter = Counter(str((record.get("target") or {}).get("move_token")) for record in records)
    if len(move_counter) < 16:
        warnings.append("manifest covers fewer than 16 unique move tokens; use only for smoke tests")
    if not action_tokens:
        warnings.append("no SO-101 action tokens were found; action-token training cannot run")
    return warnings
