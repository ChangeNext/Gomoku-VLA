from __future__ import annotations

import re
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from board import GomokuBoard

from .encoding import action_to_index
from .inference import MovePrediction


_MOVE_RE = re.compile(r"^\s*(?:BESTMOVE\s+)?(-?\d+)\s*[, ]\s*(-?\d+)", re.IGNORECASE)
_IGNORED_PREFIXES = ("INFO", "MESSAGE", "DEBUG", "SUGGEST", "FORBID")
_PISKVORK_RULE_IDS = {
    "free": 0,
    "renju": 4,
}


@dataclass(frozen=True)
class ExternalEngineConfig:
    command: str
    board_size: int = 15
    win_length: int = 5
    rule_set: str = "renju"
    enforce_center_opening: bool = True
    timeout_turn_ms: int = 1000
    protocol_timeout_s: float = 5.0
    name: str = "external_engine"


class PiskvorkEnginePolicy:
    """Move predictor for Gomocup/Piskvork-compatible Gomoku engines.

    Rapfi, Embryo, Yixin-style engines communicate over stdin/stdout. The
    protocol uses x,y coordinates, so this adapter converts to repository
    row,col coordinates and keeps board legality as the final authority.
    """

    def __init__(self, config: ExternalEngineConfig) -> None:
        self.config = config
        if self.config.board_size <= 0:
            raise ValueError("board_size must be positive")
        if self.config.timeout_turn_ms <= 0:
            raise ValueError("timeout_turn_ms must be positive")
        if self.config.protocol_timeout_s <= 0.0:
            raise ValueError("protocol_timeout_s must be positive")
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str] | None = None
        self._stdout_thread: threading.Thread | None = None

    @property
    def board_size(self) -> int:
        return self.config.board_size

    @property
    def win_length(self) -> int:
        return self.config.win_length

    @property
    def rule_set(self) -> str:
        return self.config.rule_set

    @property
    def enforce_center_opening(self) -> bool:
        return self.config.enforce_center_opening

    @property
    def name(self) -> str:
        return self.config.name

    def __enter__(self) -> "PiskvorkEnginePolicy":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        args = shlex.split(self.config.command)
        if not args:
            raise ValueError("engine command must not be empty")
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_queue = queue.Queue()
        self._stdout_thread = threading.Thread(
            target=self._read_stdout_lines,
            args=(self._process,),
            daemon=True,
        )
        self._stdout_thread.start()
        self._send(f"START {self.board_size}")
        while True:
            response = self._read_line()
            upper_response = response.upper()
            if upper_response.startswith("OK"):
                break
            if upper_response.startswith(_IGNORED_PREFIXES):
                continue
            raise RuntimeError(f"engine did not accept START {self.board_size}: {response}")
        self._send(f"INFO rule {_piskvork_rule_id(self.rule_set)}")
        self._send(f"INFO timeout_turn {self.config.timeout_turn_ms}")

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                self._write(process.stdin, "END")
            except Exception:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self._stdout_queue = None
        self._stdout_thread = None

    def new_board(
        self,
        *,
        win_length: int | None = None,
        rule_set: str | None = None,
        enforce_center_opening: bool | None = None,
    ) -> GomokuBoard:
        return GomokuBoard(
            size=self.board_size,
            win_length=self.win_length if win_length is None else win_length,
            rule_set=rule_set or self.rule_set,
            enforce_center_opening=(
                self.enforce_center_opening if enforce_center_opening is None else enforce_center_opening
            ),
        )

    def predict(self, board: GomokuBoard) -> MovePrediction:
        if board.size != self.board_size:
            raise ValueError(f"engine board_size={self.board_size} cannot predict board size {board.size}")
        if board.winner is not None:
            raise ValueError("cannot predict a move for a finished game")
        self.start()
        self._send("BOARD")
        for row, values in enumerate(board.grid):
            for col, value in enumerate(values):
                if value:
                    self._send(f"{col},{row},{int(value)}")
        self._send("DONE")
        row, col = self._read_move()
        if not board.is_legal_move(row, col):
            raise ValueError(f"engine selected illegal move: row={row}, col={col}")
        action_index = action_to_index(row, col, board.size)
        policy = np.zeros(board.size * board.size, dtype=np.float32)
        policy[action_index] = 1.0
        return MovePrediction(
            row=row,
            col=col,
            action_index=action_index,
            policy=policy,
            value=0.0,
            used_tactical_move=False,
        )

    def _read_move(self) -> tuple[int, int]:
        while True:
            line = self._read_line()
            if line.upper().startswith(_IGNORED_PREFIXES):
                continue
            match = _MOVE_RE.match(line)
            if match is None:
                raise RuntimeError(f"engine returned unrecognized move line: {line}")
            col = int(match.group(1))
            row = int(match.group(2))
            if row < 0 or col < 0:
                raise RuntimeError(f"engine returned negative move coordinates: {line}")
            return row, col

    def _send(self, line: str) -> None:
        if self._process is None:
            raise RuntimeError("engine process is not started")
        self._write(self._process.stdin, line)

    def _write(self, stream: TextIO | None, line: str) -> None:
        if stream is None:
            raise RuntimeError("engine stdin is closed")
        stream.write(line + "\n")
        stream.flush()

    def _read_line(self) -> str:
        process = self._process
        line_queue = self._stdout_queue
        if process is None or line_queue is None:
            raise RuntimeError("engine process is not started")
        deadline = time.monotonic() + self.config.protocol_timeout_s
        while True:
            if process.poll() is not None:
                stderr = ""
                if process.stderr is not None:
                    stderr = process.stderr.read().strip()
                raise RuntimeError(f"engine exited with code {process.returncode}: {stderr}")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"engine did not respond within {self.config.protocol_timeout_s:.1f}s")
            try:
                line = line_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            stripped = line.strip()
            if stripped:
                return stripped

    def _read_stdout_lines(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None or self._stdout_queue is None:
            return
        for line in process.stdout:
            self._stdout_queue.put(line)


def build_piskvork_policy(
    command: str,
    *,
    board_size: int = 15,
    win_length: int = 5,
    rule_set: str = "renju",
    enforce_center_opening: bool = True,
    timeout_turn_ms: int = 1000,
    protocol_timeout_s: float = 5.0,
    name: str | None = None,
) -> PiskvorkEnginePolicy:
    engine_name = name or Path(shlex.split(command)[0]).stem
    return PiskvorkEnginePolicy(
        ExternalEngineConfig(
            command=command,
            board_size=board_size,
            win_length=win_length,
            rule_set=rule_set,
            enforce_center_opening=enforce_center_opening,
            timeout_turn_ms=timeout_turn_ms,
            protocol_timeout_s=protocol_timeout_s,
            name=engine_name,
        )
    )


def _piskvork_rule_id(rule_set: str) -> int:
    try:
        return _PISKVORK_RULE_IDS[rule_set]
    except KeyError as exc:
        raise ValueError(f"unsupported Piskvork rule_set: {rule_set}") from exc
