from __future__ import annotations

import os
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gomoku-vla")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from board import Player
from simulation import GomokuMujocoEnv


class InteractiveGomokuApp:
    def __init__(self, robot_model: str = "so101") -> None:
        self.env = GomokuMujocoEnv(robot_model=robot_model)
        self.robot_model = robot_model
        self.human_player: Player | None = None
        self.game_over_popup_shown = False
        self.fig: Figure
        self.board_ax: Axes
        self.render_ax: Axes
        self.fig, (self.board_ax, self.render_ax) = plt.subplots(
            1,
            2,
            figsize=(13, 6.8),
            gridspec_kw={"width_ratios": [1.0, 1.25]},
        )
        self.fig.patch.set_facecolor("#d8d1c4")
        self.fig.subplots_adjust(left=0.045, right=0.985, bottom=0.11, top=0.93, wspace=0.08)
        self.fig.canvas.manager.set_window_title("Gomoku-VLA Human Play + Franka View")
        self.status = self.fig.text(0.045, 0.045, "", ha="left", va="center", fontsize=11)
        reset_ax = self.fig.add_axes((0.875, 0.027, 0.095, 0.045))
        self.reset_button = Button(reset_ax, "Reset", color="#ece6dc", hovercolor="#d9cdbc")
        self.reset_button.on_clicked(self.on_reset)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.choice_popup_buttons: list[Button] = []
        self.result_popup_buttons: list[Button] = []
        self.draw()

    def run(self) -> None:
        self.prompt_player_choice()
        plt.show()

    def prompt_player_choice(self) -> None:
        if self.human_player is not None:
            return

        popup = plt.figure(figsize=(4.1, 2.0))
        popup.patch.set_facecolor("#d8d1c4")
        popup.canvas.manager.set_window_title("Choose Stone Color")
        popup.text(0.5, 0.68, "Choose your stone color", ha="center", va="center", fontsize=13)
        black_ax = popup.add_axes((0.17, 0.24, 0.28, 0.22))
        white_ax = popup.add_axes((0.55, 0.24, 0.28, 0.22))
        black_button = Button(black_ax, "Black", color="#ece6dc", hovercolor="#d9cdbc")
        white_button = Button(white_ax, "White", color="#ece6dc", hovercolor="#d9cdbc")
        self.choice_popup_buttons = [black_button, white_button]

        def choose(player: Player) -> None:
            self.choose_player(player)
            plt.close(popup)

        black_button.on_clicked(lambda _event: choose(Player.BLACK))
        white_button.on_clicked(lambda _event: choose(Player.WHITE))
        popup.show()
        self.raise_figure(popup)
        while plt.fignum_exists(popup.number) and self.human_player is None:
            plt.pause(0.05)

    def raise_figure(self, fig: Figure) -> None:
        manager = fig.canvas.manager
        if manager is not None:
            manager.show()
        window = getattr(manager, "window", None)
        if window is None:
            fig.canvas.flush_events()
            return

        if hasattr(window, "attributes"):
            try:
                window.attributes("-topmost", True)
                window.after(500, lambda: window.attributes("-topmost", False))
            except Exception:
                pass
        for method_name in ("lift", "focus_force", "raise_", "activateWindow", "Raise"):
            method = getattr(window, method_name, None)
            if method is not None:
                try:
                    method()
                except Exception:
                    pass
        fig.canvas.flush_events()

    def on_click(self, event: object) -> None:
        if event.inaxes != self.board_ax or event.xdata is None or event.ydata is None:
            return
        if self.human_player is None:
            self.status.set_text("Choose Black or White first")
            self.fig.canvas.draw_idle()
            self.prompt_player_choice()
            return
        if self.env.board.current_player != self.human_player:
            self.status.set_text("Robot turn")
            self.fig.canvas.draw_idle()
            return
        row = self.env.board.size - 1 - round(event.ydata)
        col = round(event.xdata)
        if not self.env.board.is_on_board(row, col):
            return
        try:
            self.env.step((row, col), update_robot_target=False)
        except ValueError as exc:
            self.status.set_text(str(exc))
            self.fig.canvas.draw_idle()
            return
        self.play_robot_turn()
        self.draw()

    def on_reset(self, _event: object) -> None:
        self.env.reset()
        self.game_over_popup_shown = False
        if self.human_player == Player.WHITE:
            self.play_robot_turn()
        self.draw()

    def choose_player(self, player: Player) -> None:
        self.human_player = player
        self.env.reset()
        self.game_over_popup_shown = False
        if player == Player.WHITE:
            self.play_robot_turn()
        self.draw()

    def play_robot_turn(self) -> None:
        if (
            self.human_player is None
            or self.env.board.winner is not None
            or self.env.board.current_player == self.human_player
        ):
            return
        move = self.select_robot_move()
        if move is not None:
            if self.robot_model == "panda":
                self.env.move_panda_to_cell(*move)
            elif self.robot_model == "so101":
                self.env.move_so101_to_cell(*move)
            self.env.step(move, update_robot_target=True)

    def select_robot_move(self) -> tuple[int, int] | None:
        legal_moves = self.env.board.legal_moves()
        if not legal_moves:
            return None
        center = (self.env.board.size - 1) / 2
        return min(legal_moves, key=lambda move: (move[0] - center) ** 2 + (move[1] - center) ** 2)

    def draw(self) -> None:
        self.draw_clickable_board()
        self.draw_mujoco_render()
        winner = self.env.board.winner
        if winner == Player.EMPTY:
            self.status.set_text("Draw")
        elif winner is not None:
            self.status.set_text(self.result_message())
        elif self.human_player is None:
            self.status.set_text("Choose Black or White")
        elif self.env.board.current_player == self.human_player:
            self.status.set_text(f"Your turn: {self.human_player.name}")
        else:
            self.status.set_text(f"Robot turn: {self.env.board.current_player.name}")
        self.fig.canvas.draw_idle()
        self.show_game_over_popup()

    def show_game_over_popup(self) -> None:
        winner = self.env.board.winner
        if self.game_over_popup_shown or self.human_player is None or winner is None:
            return

        self.game_over_popup_shown = True
        result_text = self.result_message()

        popup = plt.figure(figsize=(3.6, 1.8))
        popup.patch.set_facecolor("#d8d1c4")
        popup.canvas.manager.set_window_title("Game Result")
        popup.text(0.5, 0.64, result_text, ha="center", va="center", fontsize=18, weight="bold")
        ok_ax = popup.add_axes((0.36, 0.20, 0.28, 0.22))
        ok_button = Button(ok_ax, "OK", color="#ece6dc", hovercolor="#d9cdbc")
        self.result_popup_buttons = [ok_button]
        ok_button.on_clicked(lambda _event: plt.close(popup))
        popup.show()
        self.raise_figure(popup)

    def result_message(self) -> str:
        winner = self.env.board.winner
        if winner == Player.EMPTY:
            return "Draw"
        if winner == self.human_player:
            return "You Win"
        return "You Lose"

    def draw_clickable_board(self) -> None:
        board = self.env.board
        self.board_ax.clear()
        self.board_ax.set_title("Player Board", fontsize=13, pad=10)
        self.board_ax.set_xlim(-0.5, board.size - 0.5)
        self.board_ax.set_ylim(-0.5, board.size - 0.5)
        self.board_ax.set_aspect("equal")
        self.board_ax.set_xticks(range(board.size))
        self.board_ax.set_yticks(range(board.size))
        self.board_ax.set_xticklabels([str(idx) for idx in range(board.size)], fontsize=8)
        self.board_ax.set_yticklabels([str(board.size - 1 - idx) for idx in range(board.size)], fontsize=8)
        self.board_ax.grid(color="#2a2118", linewidth=1.0)
        self.board_ax.set_facecolor("#d49a4a")
        for spine in self.board_ax.spines.values():
            spine.set_linewidth(1.6)
            spine.set_color("#20150d")

        for row, values in enumerate(board.grid):
            for col, value in enumerate(values):
                if value == Player.EMPTY:
                    continue
                color = "black" if value == Player.BLACK else "white"
                edgecolor = "black"
                visual_y = board.size - 1 - row
                self.board_ax.scatter(col, visual_y, s=220, c=color, edgecolors=edgecolor, linewidths=1.2, zorder=3)

    def draw_mujoco_render(self) -> None:
        self.render_ax.clear()
        self.render_ax.set_title("Franka Robot View", fontsize=13, pad=10)
        self.render_ax.imshow(self.env.render(width=980, height=720, camera="robot_full"))
        self.render_ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-model", choices=("kinematic", "panda", "so101"), default="so101")
    args = parser.parse_args()
    InteractiveGomokuApp(robot_model=args.robot_model).run()


if __name__ == "__main__":
    main()
