from __future__ import annotations

import argparse
import queue
import time

import glfw
import mujoco
import mujoco.viewer

from simulation import GomokuMujocoEnv


class ViewerController:
    def __init__(self, dual_view: bool = True) -> None:
        self.env = GomokuMujocoEnv()
        self.dual_view = dual_view
        self.key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
        self.handles: list[mujoco.viewer.Handle] = []

    def run(self) -> None:
        self.handles.append(self._launch_viewer("iso"))
        if self.dual_view:
            self.handles.append(self._launch_viewer("top"))
        self._sync_all()
        while any(handle.is_running() for handle in self.handles):
            self._drain_input()
            self._sync_all(state_only=False)
            time.sleep(1.0 / 60.0)
        for handle in self.handles:
            handle.close()

    def _launch_viewer(self, camera_name: str) -> mujoco.viewer.Handle:
        handle = mujoco.viewer.launch_passive(
            self.env.model,
            self.env.data,
            key_callback=self._queue_key,
            show_left_ui=False,
            show_right_ui=False,
        )
        with handle.lock():
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            handle.cam.fixedcamid = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
        return handle

    def _queue_key(self, key: int) -> None:
        self.key_events.put(key)

    def _drain_input(self) -> None:
        while not self.key_events.empty():
            self._handle_key(self.key_events.get_nowait())

    def _handle_key(self, key: int) -> None:
        if key in (glfw.KEY_LEFT, glfw.KEY_A, glfw.KEY_H):
            self.env.move_selection(0, -1)
        elif key in (glfw.KEY_RIGHT, glfw.KEY_D, glfw.KEY_L):
            self.env.move_selection(0, 1)
        elif key in (glfw.KEY_UP, glfw.KEY_W, glfw.KEY_K):
            self.env.move_selection(-1, 0)
        elif key in (glfw.KEY_DOWN, glfw.KEY_S, glfw.KEY_J):
            self.env.move_selection(1, 0)
        elif key in (glfw.KEY_ENTER, glfw.KEY_SPACE):
            try:
                self.env.place_selected()
            except ValueError:
                pass
        elif key in (glfw.KEY_BACKSPACE, glfw.KEY_R):
            self.env.reset()

    def _sync_all(self, state_only: bool = False) -> None:
        left, right = self.env.status_lines()
        texts = (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "arrows / wasd move\nspace place\nr reset",
            f"{left}\n{right}",
        )
        for handle in self.handles:
            if not handle.is_running():
                continue
            handle.set_texts(texts)
            handle.sync(state_only=state_only)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-view", action="store_true")
    args = parser.parse_args()
    ViewerController(dual_view=not args.single_view).run()


if __name__ == "__main__":
    main()
