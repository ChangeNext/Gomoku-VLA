from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gomoku_ai.train import plot_training_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AlphaZero training history from a CSV file.")
    parser.add_argument("history_csv")
    parser.add_argument("--output", default="gomoku_ai/runs/adhoc/plots/training.png")
    args = parser.parse_args()

    with Path(args.history_csv).open(newline="") as csv_file:
        history = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(csv_file)
        ]
    plot_training_history(history, Path(args.output))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
