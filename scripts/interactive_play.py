from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gomoku-vla")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from board import Player
from simulation import GomokuMujocoEnv


class InteractiveGomokuApp:
    def __init__(self) -> None:
        self.env = GomokuMujocoEnv()
        self.fig: Figure
        self.board_ax: Axes
        self.render_ax: Axes
        self.fig, (self.board_ax, self.render_ax) = plt.subplots(
            1,
            2,
            figsize=(13, 6.8),
            gridspec_kw={"width_ratios": [1.0, 1.25]},
        )
        self.fig.patch.set_facecolor("#f4f1ea")
        self.fig.subplots_adjust(left=0.045, right=0.985, bottom=0.11, top=0.93, wspace=0.08)
        self.fig.canvas.manager.set_window_title("Gomoku-VLA Interactive MuJoCo")
        self.status = self.fig.text(0.045, 0.045, "", ha="left", va="center", fontsize=11)
        reset_ax = self.fig.add_axes((0.875, 0.027, 0.095, 0.045))
        self.reset_button = Button(reset_ax, "Reset", color="#e5dfd2", hovercolor="#d8cfbd")
        self.reset_button.on_clicked(self.on_reset)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.draw()

    def run(self) -> None:
        plt.show()

    def on_click(self, event: object) -> None:
        if event.inaxes != self.board_ax or event.xdata is None or event.ydata is None:
            return
        row = self.env.board.size - 1 - round(event.ydata)
        col = round(event.xdata)
        if not self.env.board.is_on_board(row, col):
            return
        try:
            self.env.step((row, col))
        except ValueError as exc:
            self.status.set_text(str(exc))
        self.draw()

    def on_reset(self, _event: object) -> None:
        self.env.reset()
        self.draw()

    def draw(self) -> None:
        self.draw_clickable_board()
        self.draw_mujoco_render()
        winner = self.env.board.winner
        if winner == Player.EMPTY:
            self.status.set_text("Draw")
        elif winner is not None:
            self.status.set_text(f"Winner: {winner.name}")
        else:
            self.status.set_text(f"Turn: {self.env.board.current_player.name}")
        self.fig.canvas.draw_idle()

    def draw_clickable_board(self) -> None:
        board = self.env.board
        self.board_ax.clear()
        self.board_ax.set_title("Human board input", fontsize=13, pad=10)
        self.board_ax.set_xlim(-0.5, board.size - 0.5)
        self.board_ax.set_ylim(-0.5, board.size - 0.5)
        self.board_ax.set_aspect("equal")
        self.board_ax.set_xticks(range(board.size))
        self.board_ax.set_yticks(range(board.size))
        self.board_ax.set_xticklabels([str(idx) for idx in range(board.size)], fontsize=8)
        self.board_ax.set_yticklabels([str(board.size - 1 - idx) for idx in range(board.size)], fontsize=8)
        self.board_ax.grid(color="#222222", linewidth=1.0)
        self.board_ax.set_facecolor("#e7b264")
        for spine in self.board_ax.spines.values():
            spine.set_linewidth(1.6)
            spine.set_color("#19140c")

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
        self.render_ax.set_title("MuJoCo Panda-like robot + board", fontsize=13, pad=10)
        self.render_ax.imshow(self.env.render(width=900, height=720, camera="iso"))
        self.render_ax.axis("off")


def main() -> None:
    InteractiveGomokuApp().run()


if __name__ == "__main__":
    main()
