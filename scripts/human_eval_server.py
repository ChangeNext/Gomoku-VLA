from __future__ import annotations

import argparse
import csv
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from board import GomokuBoard, Player


SERVER_VERSION = "human-eval-2026-07-03-csv-simple-board"


class MoveSelector(Protocol):
    def __call__(self, board: GomokuBoard) -> tuple[int, int]:
        ...


@dataclass
class HumanEvalConfig:
    checkpoint: str
    output_jsonl: str | None = None
    output_csv: str | None = None
    simulations: int = 32
    win_length: int = 5
    rule_set: str | None = None
    enforce_center_opening: bool | None = None
    device: str = "auto"


@dataclass
class MoveRecord:
    player: str
    row: int
    col: int


@dataclass
class EvalGame:
    game_id: str
    player_id: str
    checkpoint: str
    board: GomokuBoard
    human_player: Player
    ai_player: Player
    started_at: str
    moves: list[MoveRecord] = field(default_factory=list)
    ended_at: str | None = None
    result: str | None = None


class HumanEvalStore:
    def __init__(self, config: HumanEvalConfig, ai_move_selector: MoveSelector) -> None:
        self.config = config
        self._ai_move_selector = ai_move_selector
        self._games: dict[str, EvalGame] = {}
        self._lock = threading.Lock()
        self.output_jsonl_path, self.output_csv_path = resolve_output_paths(config)

    def new_game(self, player_id: str, human_color: str) -> dict[str, Any]:
        human_player = _parse_player(human_color)
        board = GomokuBoard(
            size=self.board_size,
            win_length=self.config.win_length,
            rule_set=self.rule_set,
            enforce_center_opening=self.enforce_center_opening,
        )
        game = EvalGame(
            game_id=str(uuid.uuid4()),
            player_id=player_id.strip() or "anonymous",
            checkpoint=self.config.checkpoint,
            board=board,
            human_player=human_player,
            ai_player=human_player.opponent,
            started_at=_now_iso(),
        )
        if board.current_player == game.ai_player:
            self._place_ai_move(game)
        with self._lock:
            self._games[game.game_id] = game
        return self._public_state(game)

    def human_move(self, game_id: str, row: int, col: int) -> dict[str, Any]:
        with self._lock:
            game = self._games.get(game_id)
            if game is None:
                raise ValueError(f"unknown game_id: {game_id}")
            if game.board.winner is not None:
                return self._public_state(game)
            if game.board.current_player != game.human_player:
                raise ValueError("it is not the human player's turn")
            game.board.place(row, col)
            game.moves.append(MoveRecord(player=game.human_player.name.lower(), row=row, col=col))
            self._finish_if_done(game)
            if game.board.winner is None:
                self._place_ai_move(game)
                self._finish_if_done(game)
            return self._public_state(game)

    def stats(self) -> dict[str, Any]:
        path = self.output_jsonl_path
        totals = {"games": 0, "human_win": 0, "ai_win": 0, "draw": 0}
        by_color = {
            "black": {"games": 0, "human_win": 0, "ai_win": 0, "draw": 0},
            "white": {"games": 0, "human_win": 0, "ai_win": 0, "draw": 0},
        }
        if not path.exists():
            return {"totals": totals, "by_human_color": by_color}

        with path.open(encoding="utf-8") as log_file:
            for line in log_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                result = record.get("result")
                color = record.get("human_color")
                if result not in {"human_win", "ai_win", "draw"}:
                    continue
                totals["games"] += 1
                totals[result] += 1
                if color in by_color:
                    by_color[color]["games"] += 1
                    by_color[color][result] += 1
        return {"totals": totals, "by_human_color": by_color}

    @property
    def board_size(self) -> int:
        return int(getattr(self._ai_move_selector, "board_size", 15))

    @property
    def rule_set(self) -> str:
        selector_rule_set = getattr(self._ai_move_selector, "rule_set", "free")
        return self.config.rule_set or str(selector_rule_set)

    @property
    def enforce_center_opening(self) -> bool:
        selector_value = bool(getattr(self._ai_move_selector, "enforce_center_opening", False))
        return selector_value if self.config.enforce_center_opening is None else self.config.enforce_center_opening

    def _place_ai_move(self, game: EvalGame) -> None:
        if game.board.current_player != game.ai_player:
            return
        row, col = self._ai_move_selector(game.board)
        game.board.place(row, col)
        game.moves.append(MoveRecord(player=game.ai_player.name.lower(), row=row, col=col))

    def _finish_if_done(self, game: EvalGame) -> None:
        if game.board.winner is None or game.result is not None:
            return
        game.ended_at = _now_iso()
        if game.board.winner == Player.EMPTY:
            game.result = "draw"
        elif game.board.winner == game.human_player:
            game.result = "human_win"
        else:
            game.result = "ai_win"
        self._append_finished_game(game)

    def _append_finished_game(self, game: EvalGame) -> None:
        path = self.output_jsonl_path
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "game_id": game.game_id,
            "player_id": game.player_id,
            "checkpoint": game.checkpoint,
            "human_color": game.human_player.name.lower(),
            "ai_color": game.ai_player.name.lower(),
            "result": game.result,
            "move_count": game.board.move_count,
            "moves": [asdict(move) for move in game.moves],
            "started_at": game.started_at,
            "ended_at": game.ended_at,
            "board_size": game.board.size,
            "win_length": game.board.win_length,
            "rule_set": game.board.rule_set,
            "enforce_center_opening": game.board.enforce_center_opening,
            "simulations": self.config.simulations,
        }
        with path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")
        self._append_finished_game_csv(game)

    def _append_finished_game_csv(self, game: EvalGame) -> None:
        path = self.output_csv_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "game_id",
            "player_id",
            "checkpoint",
            "human_color",
            "ai_color",
            "result",
            "move_count",
            "started_at",
            "ended_at",
            "board_size",
            "win_length",
            "rule_set",
            "simulations",
            "moves",
        ]
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "game_id": game.game_id,
                    "player_id": game.player_id,
                    "checkpoint": game.checkpoint,
                    "human_color": game.human_player.name.lower(),
                    "ai_color": game.ai_player.name.lower(),
                    "result": game.result,
                    "move_count": game.board.move_count,
                    "started_at": game.started_at,
                    "ended_at": game.ended_at,
                    "board_size": game.board.size,
                    "win_length": game.board.win_length,
                    "rule_set": game.board.rule_set,
                    "simulations": self.config.simulations,
                    "moves": " ".join(f"{move.player}:{move.row},{move.col}" for move in game.moves),
                }
            )

    def _public_state(self, game: EvalGame) -> dict[str, Any]:
        return {
            "game_id": game.game_id,
            "player_id": game.player_id,
            "board": game.board.copy_state(),
            "board_size": game.board.size,
            "current_player": game.board.current_player.name.lower(),
            "human_color": game.human_player.name.lower(),
            "ai_color": game.ai_player.name.lower(),
            "winner": None if game.board.winner is None else game.board.winner.name.lower(),
            "result": game.result,
            "move_count": game.board.move_count,
            "moves": [asdict(move) for move in game.moves],
        }


class CheckpointMoveSelector:
    def __init__(self, config: HumanEvalConfig) -> None:
        from gomoku_ai.mcts import MCTSConfig
        from gomoku_ai.self_play import select_greedy_move
        from gomoku_ai.torch_model import TorchPolicyValueModel, load_checkpoint
        from scripts.play_ai_cli import resolve_device

        device = resolve_device(config.device)
        network = load_checkpoint(config.checkpoint, device=device)
        self.board_size = network.board_size
        self.rule_set = network.rule_set
        self.enforce_center_opening = network.enforce_center_opening
        self._model = TorchPolicyValueModel(network, device=device)
        self._mcts_config = MCTSConfig(simulations=config.simulations, temperature=0.0)
        self._select_greedy_move = select_greedy_move

    def __call__(self, board: GomokuBoard) -> tuple[int, int]:
        return self._select_greedy_move(board, self._model, self._mcts_config)


def create_app(config: HumanEvalConfig):
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Install web dependencies with: pip install -e \".[learning,web]\"") from exc

    app = FastAPI(title="Gomoku Human Evaluation")
    store = HumanEvalStore(config, CheckpointMoveSelector(config))

    @app.post("/api/new-game")
    async def new_game(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        payload = payload or {}
        try:
            return store.new_game(str(payload.get("player_id", "anonymous")), str(payload.get("human_color", "black")))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/move")
    async def move(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        payload = payload or {}
        try:
            return store.human_move(str(payload["game_id"]), int(payload["row"]), int(payload["col"]))
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"missing field: {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return store.stats()

    @app.get("/api/version")
    def version() -> dict[str, str]:
        return {"version": SERVER_VERSION}

    web_dir = Path(__file__).resolve().parents[1] / "web"
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def _parse_player(raw: str) -> Player:
    value = raw.strip().lower()
    if value == "black":
        return Player.BLACK
    if value == "white":
        return Player.WHITE
    raise ValueError("human_color must be black or white")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_output_paths(config: HumanEvalConfig) -> tuple[Path, Path]:
    checkpoint = Path(config.checkpoint)
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    evaluation_dir = run_dir / "evaluation"
    stem = checkpoint.stem
    jsonl_path = Path(config.output_jsonl) if config.output_jsonl else evaluation_dir / f"{stem}_human_eval.jsonl"
    csv_path = Path(config.output_csv) if config.output_csv else evaluation_dir / f"{stem}_human_eval.csv"
    return jsonl_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a browser UI for human evaluation against a best checkpoint.")
    parser.add_argument("--checkpoint", default="gomoku_ai/runs/alphazero_9x9_long/checkpoints/best.pt")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--rule-set", choices=("free", "renju"))
    parser.add_argument("--center-opening", action="store_true")
    parser.add_argument("--no-center-opening", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

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

    config = HumanEvalConfig(
        checkpoint=str(checkpoint),
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
        simulations=args.simulations,
        win_length=args.win_length,
        rule_set=args.rule_set,
        enforce_center_opening=enforce_center_opening,
        device=args.device,
    )
    app = create_app(config)

    import uvicorn

    print(f"Starting Gomoku human eval server: {SERVER_VERSION}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
