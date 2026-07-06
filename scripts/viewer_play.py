from __future__ import annotations

import argparse
import queue
import time

import glfw
import mujoco
import mujoco.viewer

from simulation import GomokuMujocoEnv


class ViewerController:
    def __init__(self, dual_view: bool = True, robot_model: str = "kinematic") -> None:
        self.env = GomokuMujocoEnv(robot_model=robot_model)
        self.dual_view = dual_view
        self.robot_model = robot_model
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
            self._lock_viewer_visuals(handle)
        return handle

    def _queue_key(self, key: int) -> None:
        self.key_events.put(key)

    def _drain_input(self) -> None:
        while not self.key_events.empty():
            self._handle_key(self.key_events.get_nowait())

    def _handle_key(self, key: int) -> None:
        moved = False
        if key in (glfw.KEY_LEFT, glfw.KEY_A, glfw.KEY_H):
            self.env.move_selection(0, -1)
            moved = True
        elif key in (glfw.KEY_RIGHT, glfw.KEY_D, glfw.KEY_L):
            self.env.move_selection(0, 1)
            moved = True
        elif key in (glfw.KEY_UP, glfw.KEY_W, glfw.KEY_K):
            self.env.move_selection(-1, 0)
            moved = True
        elif key in (glfw.KEY_DOWN, glfw.KEY_S, glfw.KEY_J):
            self.env.move_selection(1, 0)
            moved = True
        elif key in (glfw.KEY_ENTER, glfw.KEY_SPACE):
            try:
                if self.robot_model == "panda":
                    self._place_selected_with_panda()
                elif self.robot_model == "so101":
                    self._place_selected_with_so101()
                else:
                    self.env.place_selected()
            except ValueError:
                pass
        elif key in (glfw.KEY_BACKSPACE, glfw.KEY_R):
            self.env.reset()
        if moved and self.robot_model == "panda":
            self._animate_panda_to_selected_cell(include_gripper=False, steps=35)
        elif moved and self.robot_model == "so101":
            self._animate_so101_to_selected_cell(include_gripper=False, steps=30)

    def _animate_panda_to_selected_cell(self, *, include_gripper: bool, steps: int) -> None:
        target_xyz = self.env.panda_target_pose_for_cell(*self.env.selected_cell)
        self._animate_panda_to_target(target_xyz, gripper=1.0, steps=steps)
        if not include_gripper:
            return
        for gripper in (0.0, 1.0):
            self.env.set_panda_gripper(gripper)
            self.env.simulate(30)
            self._sync_all(state_only=False)
            time.sleep(0.12)

    def _animate_panda_to_target(self, target_xyz: tuple[float, float, float], *, gripper: float, steps: int) -> None:
        joint_targets = self.env.solve_panda_ik(target_xyz)
        trajectory = self.env.interpolate_panda_joint_trajectory(joint_targets, steps=steps)
        for waypoint in trajectory:
            self.env.set_panda_joint_targets(waypoint, gripper=gripper)
            self.env.simulate(6)
            self._sync_all(state_only=False)
            time.sleep(1.0 / 120.0)

    def _animate_so101_to_selected_cell(self, *, include_gripper: bool, steps: int) -> None:
        target_xyz = self.env.so101_target_pose_for_cell(*self.env.selected_cell)
        self._animate_so101_to_target(target_xyz, gripper=1.0, steps=steps)
        if not include_gripper:
            return
        for gripper in (0.0, 1.0):
            self.env.set_so101_gripper(gripper)
            self.env.simulate(30)
            self._sync_all(state_only=False)
            time.sleep(0.12)

    def _animate_so101_to_target(self, target_xyz: tuple[float, float, float], *, gripper: float, steps: int) -> None:
        joint_targets = self.env.solve_so101_ik(target_xyz)
        trajectory = self.env.interpolate_so101_joint_trajectory(joint_targets, steps=steps)
        for waypoint in trajectory:
            self.env.set_so101_joint_targets(waypoint, gripper=gripper)
            self.env.simulate(6)
            self._sync_all(state_only=False)
            time.sleep(1.0 / 120.0)

    def _place_selected_with_panda(self) -> None:
        row, col = self.env.selected_cell
        hover = self.env.panda_target_pose_for_cell(row, col, clearance=0.180)
        place = self.env.panda_target_pose_for_cell(row, col, clearance=0.060)
        self._animate_panda_to_target(hover, gripper=0.0, steps=45)
        self._animate_panda_to_target(place, gripper=0.0, steps=35)
        self.env.place_selected()
        self.env.set_panda_gripper(1.0)
        self.env.simulate(30)
        self._sync_all(state_only=False)
        time.sleep(0.12)
        self._animate_panda_to_target(hover, gripper=1.0, steps=45)

    def _place_selected_with_so101(self) -> None:
        row, col = self.env.selected_cell
        hover = self.env.so101_target_pose_for_cell(row, col, clearance=0.075)
        place = self.env.so101_target_pose_for_cell(row, col, clearance=0.025)
        self._animate_so101_to_target(hover, gripper=0.0, steps=30)
        self._animate_so101_to_target(place, gripper=0.0, steps=35)
        self.env.place_selected()
        self.env.set_so101_gripper(1.0)
        self.env.simulate(30)
        self._sync_all(state_only=False)
        time.sleep(0.12)
        self._animate_so101_to_target(hover, gripper=1.0, steps=45)

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
            with handle.lock():
                self._lock_viewer_visuals(handle)
            handle.sync(state_only=state_only)
            with handle.lock():
                self._lock_viewer_visuals(handle)

    def _lock_viewer_visuals(self, handle: mujoco.viewer.Handle) -> None:
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = 0
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1
        if handle.user_scn is not None:
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_ADDITIVE] = 0
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-view", action="store_true")
    parser.add_argument("--robot-model", choices=("kinematic", "panda", "so101"), default="kinematic")
    args = parser.parse_args()
    ViewerController(dual_view=not args.single_view, robot_model=args.robot_model).run()


if __name__ == "__main__":
    main()
