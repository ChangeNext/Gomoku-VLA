from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from scripts.human_eval_server import CheckpointMoveSelector, HumanEvalConfig, HumanEvalStore, SERVER_VERSION


class HumanEvalRequestHandler(SimpleHTTPRequestHandler):
    store: HumanEvalStore
    web_dir: Path

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(self.web_dir), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/version":
            self._send_json({"version": SERVER_VERSION})
            return
        if self.path == "/api/stats":
            self._send_json(self.store.stats())
            return
        super().do_GET()

    def do_POST(self) -> None:
        payload = self._read_json_payload()
        try:
            if self.path == "/api/new-game":
                player_id = str(payload.get("player_id", "anonymous"))
                human_color = str(payload.get("human_color", "black"))
                self._send_json(self.store.new_game(player_id, human_color))
                return
            if self.path == "/api/move":
                self._send_json(
                    self.store.human_move(
                        str(payload["game_id"]),
                        int(payload["row"]),
                        int(payload["col"]),
                    )
                )
                return
        except KeyError as exc:
            self._send_json({"detail": f"missing field: {exc.args[0]}"}, HTTPStatus.BAD_REQUEST)
            return
        except ValueError as exc:
            self._send_json({"detail": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"detail": "not found"}, HTTPStatus.NOT_FOUND)

    def _read_json_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the simple Gomoku human evaluation UI.")
    parser.add_argument("--checkpoint", required=True)
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
    print(f"Loading checkpoint before serving: {checkpoint}", flush=True)
    selector = CheckpointMoveSelector(config)
    print("Checkpoint loaded. Starting web server.", flush=True)
    HumanEvalRequestHandler.store = HumanEvalStore(config, selector)
    HumanEvalRequestHandler.web_dir = Path(__file__).resolve().parents[1] / "web"

    server = ThreadingHTTPServer((args.host, args.port), HumanEvalRequestHandler)
    print(f"Starting simple Gomoku human eval server: {SERVER_VERSION}", flush=True)
    print(f"Writing JSONL: {HumanEvalRequestHandler.store.output_jsonl_path}", flush=True)
    print(f"Writing CSV:   {HumanEvalRequestHandler.store.output_csv_path}", flush=True)
    print(f"Serving http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
