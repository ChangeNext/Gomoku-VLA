from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_shards(inputs: list[Path], output: Path) -> tuple[int, int]:
    records: list[dict[str, object]] = []
    game_ids: set[str] = set()
    steps: set[tuple[str, int]] = set()
    for source in inputs:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
                game_id = str(record.get("game_id", ""))
                step = int(record.get("step", -1))
                key = (game_id, step)
                if not game_id or key in steps:
                    raise ValueError(f"missing or duplicate game/step in shards: {key}")
                steps.add(key)
                game_ids.add(game_id)
                records.append(record)
    records.sort(key=lambda record: (int((record.get("observation") or {}).get("episode_index", -1)), int(record["step"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(game_ids), len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent SO-101 collection JSONL shards.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    games, records = merge_shards([Path(value) for value in args.inputs], Path(args.output))
    print(f"merged {records} records from {games} games into {args.output}", flush=True)


if __name__ == "__main__":
    main()
