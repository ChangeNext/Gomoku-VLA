from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ManifestSplitSummary:
    manifest_path: str
    output_dir: str
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    total_samples: int
    total_games: int
    split_samples: dict[str, int]
    split_games: dict[str, int]
    artifacts: dict[str, str]


def split_openvla_manifest_by_game(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 20260719,
    output_prefix: str = "manifest",
) -> ManifestSplitSummary:
    """Split an OpenVLA manifest by game id so related board states do not leak."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    manifest = Path(manifest_path)
    target_dir = Path(output_dir) if output_dir is not None else manifest.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    records = list(_read_jsonl(manifest))
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        game_id = ((record.get("source") or {}).get("game_id"))
        if not game_id:
            raise ValueError(f"record[{index}] is missing source.game_id")
        games[str(game_id)].append(record)
    if not games:
        raise ValueError("manifest contains no records")

    game_ids = sorted(games)
    random.Random(seed).shuffle(game_ids)
    counts = _split_counts(len(game_ids), train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    train_ids = set(game_ids[: counts["train"]])
    val_ids = set(game_ids[counts["train"] : counts["train"] + counts["val"]])
    test_ids = set(game_ids[counts["train"] + counts["val"] :])

    splits = {
        "train": [record for game_id in game_ids if game_id in train_ids for record in games[game_id]],
        "val": [record for game_id in game_ids if game_id in val_ids for record in games[game_id]],
        "test": [record for game_id in game_ids if game_id in test_ids for record in games[game_id]],
    }
    split_game_ids = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    artifacts: dict[str, str] = {}
    for name, split_records in splits.items():
        path = target_dir / f"{output_prefix}_{name}.jsonl"
        _write_jsonl(path, split_records)
        artifacts[name] = str(path)

    summary = ManifestSplitSummary(
        manifest_path=str(manifest),
        output_dir=str(target_dir),
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        total_samples=len(records),
        total_games=len(games),
        split_samples={name: len(split_records) for name, split_records in splits.items()},
        split_games={name: len(ids) for name, ids in split_game_ids.items()},
        artifacts=artifacts,
    )
    metadata = {
        **summary.__dict__,
        "split_game_ids": split_game_ids,
        "policy": "game_id_grouped_shuffle",
    }
    (target_dir / f"{output_prefix}_splits.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = {"train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio}
    for name, value in ratios.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if train_ratio <= 0.0:
        raise ValueError("train_ratio must be positive")


def _split_counts(
    total_games: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    if total_games <= 0:
        raise ValueError("total_games must be positive")
    test_count = int(round(total_games * test_ratio))
    val_count = int(round(total_games * val_ratio))
    if test_ratio > 0.0 and total_games >= 3:
        test_count = max(1, test_count)
    if val_ratio > 0.0 and total_games >= 3:
        val_count = max(1, val_count)
    if val_count + test_count >= total_games:
        overflow = val_count + test_count - total_games + 1
        reduce_test = min(test_count, overflow)
        test_count -= reduce_test
        overflow -= reduce_test
        val_count = max(0, val_count - overflow)
    train_count = total_games - val_count - test_count
    return {"train": train_count, "val": val_count, "test": test_count}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
