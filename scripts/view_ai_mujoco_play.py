from __future__ import annotations

import argparse
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer

from board import Player
from gomoku_ai.inference import CheckpointPolicy
from robot_control import RobotSafetyController
from simulation import GomokuMujocoEnv, build_pick_place_action


class AIMujocoViewer:
    def __init__(
        self,
        checkpoint: Path,
        *,
        simulations: int,
        device: str,
        win_length: int,
        rule_set: str | None,
        enforce_center_opening: bool | None,
        max_moves: int | None,
        camera: str,
        move_delay: float,
        waypoint_delay: float,
        robot_model: str,
    ) -> None:
        self.policy = CheckpointPolicy(checkpoint, device=device, simulations=simulations)
        self.env = GomokuMujocoEnv(
            board_size=self.policy.board_size,
            win_length=win_length,
            rule_set=rule_set or self.policy.rule_set,
            enforce_center_opening=(
                self.policy.enforce_center_opening
                if enforce_center_opening is None
                else enforce_center_opening
            ),
            show_robot=True,
            robot_model=robot_model,
        )
        self.safety = RobotSafetyController(self.env)
        self.checkpoint = checkpoint
        self.max_moves = max_moves or self.env.board.size * self.env.board.size
        self.camera = camera
        self.move_delay = move_delay
        self.waypoint_delay = waypoint_delay
        self.robot_model = robot_model
        self.paused = False
        self.single_step_requested = False
        self.stopped = False
        self.last_status = "loaded checkpoint"
        self._active_handle: mujoco.viewer.Handle | None = None

    def run(self) -> None:
        handle = mujoco.viewer.launch_passive(
            self.env.model,
            self.env.data,
            key_callback=self._handle_key,
            show_left_ui=False,
            show_right_ui=False,
        )
        self._active_handle = handle
        try:
            self._set_camera(handle)
            self._lock_viewer_visuals(handle)
            self._sync(handle)
            while handle.is_running() and not self.stopped:
                if self.paused and not self.single_step_requested:
                    self._sync(handle)
                    self._sleep_with_visual_lock(handle, 1.0 / 30.0)
                    continue
                if self.env.board.winner is not None or self.env.board.move_count >= self.max_moves:
                    self.last_status = "finished"
                    self._sync(handle)
                    self._sleep_with_visual_lock(handle, 1.0 / 30.0)
                    continue
                single_step = self.single_step_requested
                self.single_step_requested = False
                if single_step:
                    self.paused = False
                self._play_ai_move(handle)
                if single_step:
                    self.paused = True
        finally:
            self._active_handle = None
            handle.close()

    def _play_ai_move(self, handle: mujoco.viewer.Handle) -> None:
        player = self.env.board.current_player
        prediction = self.policy.predict(self.env.board)
        row, col = prediction.move
        self.last_status = (
            f"{player.name.lower()} -> ({row}, {col}) "
            f"value={prediction.value:.3f}"
        )
        self.safety.validate_pick(player).raise_if_unsafe()
        self.safety.validate_place_cell(row, col, player).raise_if_unsafe()
        with handle.lock():
            self.env.set_selection(row, col, update_robot_target=False)
        self._sync(handle)
        self._sleep_with_visual_lock(handle, self.move_delay)

        if self.robot_model == "panda":
            self._play_panda_pick_place(handle, row, col, player)
        elif self.robot_model == "so101":
            self._play_so101_pick_place(handle, row, col, player)
        else:
            self._play_kinematic_pick_place(handle, row, col, player)

        self._sync(handle, phase="placed")
        self._sleep_with_visual_lock(handle, self.move_delay)

    def _play_kinematic_pick_place(self, handle: mujoco.viewer.Handle, row: int, col: int, player) -> None:
        robot_action = build_pick_place_action(self.env, (row, col), player)
        self.safety.validate_action_trace(robot_action).raise_if_unsafe()
        for point in robot_action["ee_trajectory"]:
            if not handle.is_running() or self.stopped:
                return
            while self.paused and handle.is_running() and not self.stopped:
                self._sync(handle, phase=str(point["phase"]))
                time.sleep(1.0 / 30.0)
            pose = point["pose"]
            with handle.lock():
                self.env.set_robot_hand_world(
                    float(pose["x"]),
                    float(pose["y"]),
                    float(pose["z"]),
                    gripper=float(point["gripper"]),
                )
                if point["phase"] == "grasp":
                    self.env.grasp_supply_stone(player, float(pose["x"]), float(pose["y"]), float(pose["z"]))
                elif point["phase"] in {"lift", "pre_place", "place"}:
                    self.env.set_held_stone_world(float(pose["x"]), float(pose["y"]), float(pose["z"]), player)
                elif point["phase"] == "release":
                    self.env.commit_held_stone_to_cell(row, col, update_robot_target=True)
                else:
                    self.env.clear_held_stone()
            self._sync(handle, phase=str(point["phase"]))
            self._sleep_with_visual_lock(handle, self.waypoint_delay)

    def _play_panda_pick_place(self, handle: mujoco.viewer.Handle, row: int, col: int, player: Player) -> None:
        pick_x, pick_y, pick_z = self.env.stone_supply_world(player)
        place_x, place_y, place_z = self.env.board_to_world(row, col)
        pick_hover = (pick_x, pick_y, pick_z + 0.180)
        pick_low = (pick_x, pick_y, pick_z + 0.070)
        place_hover = (place_x, place_y, place_z + 0.180)
        place_low = (place_x, place_y, place_z + 0.070)

        self._move_panda_to_xyz(handle, pick_hover, gripper=1.0, phase="panda_pre_pick", steps=65)
        self._move_panda_to_xyz(handle, pick_low, gripper=1.0, phase="panda_pick", steps=35)
        self._set_panda_gripper(handle, 0.0, phase="panda_grasp", held_player=player)
        self._move_panda_to_xyz(handle, pick_hover, gripper=0.0, phase="panda_lift", steps=45, held_player=player)
        self._move_panda_to_xyz(handle, place_hover, gripper=0.0, phase="panda_pre_place", steps=80, held_player=player)
        self._move_panda_to_xyz(handle, place_low, gripper=0.0, phase="panda_place", steps=35, held_player=player)
        with handle.lock():
            self.env.commit_held_stone_to_cell(row, col, update_robot_target=True)
        self._set_panda_gripper(handle, 1.0, phase="panda_release")
        self._move_panda_to_xyz(handle, place_hover, gripper=1.0, phase="panda_retreat", steps=45)

    def _play_so101_pick_place(self, handle: mujoco.viewer.Handle, row: int, col: int, player: Player) -> None:
        pick_x, pick_y, pick_z = self.env.stone_supply_world(player)
        place_x, place_y, place_z = self.env.board_to_world(row, col)
        pick_hover = (pick_x, pick_y, pick_z + 0.090)
        pick_low = (pick_x, pick_y, pick_z + 0.032)
        place_hover = (place_x, place_y, place_z + 0.090)
        place_low = (place_x, place_y, place_z + 0.032)

        self._move_so101_to_xyz(handle, pick_hover, gripper=1.0, phase="so101_pre_pick", steps=50)
        self._move_so101_to_xyz(handle, pick_low, gripper=1.0, phase="so101_pick", steps=30)
        self._set_so101_gripper(handle, 0.0, phase="so101_grasp", held_player=player)
        self._move_so101_to_xyz(handle, pick_hover, gripper=0.0, phase="so101_lift", steps=35, held_player=player)
        self._move_so101_to_xyz(handle, place_hover, gripper=0.0, phase="so101_pre_place", steps=65, held_player=player)
        self._move_so101_to_xyz(handle, place_low, gripper=0.0, phase="so101_place", steps=30, held_player=player)
        with handle.lock():
            self.env.commit_held_stone_to_cell(row, col, update_robot_target=True)
        self._set_so101_gripper(handle, 1.0, phase="so101_release")
        self._move_so101_to_xyz(handle, place_hover, gripper=1.0, phase="so101_retreat", steps=35)

    def _move_panda_to_xyz(
        self,
        handle: mujoco.viewer.Handle,
        xyz: tuple[float, float, float],
        *,
        gripper: float,
        phase: str,
        steps: int,
        held_player: Player | None = None,
    ) -> None:
        with handle.lock():
            joint_targets = self.env.solve_panda_ik(xyz)
            trajectory = self.env.interpolate_panda_joint_trajectory(joint_targets, steps=steps)
        for index, waypoint in enumerate(trajectory):
            if not handle.is_running() or self.stopped:
                return
            while self.paused and handle.is_running() and not self.stopped:
                self._sync(handle, phase=f"{phase}_paused")
                time.sleep(1.0 / 30.0)
            with handle.lock():
                self.env.set_panda_joint_targets(waypoint, gripper=gripper)
                self.env.simulate(6)
                if held_player is not None:
                    self._show_held_stone_at_gripper("panda", held_player)
            self._sync(handle, phase=f"{phase} {index + 1}/{len(trajectory)}")
            self._sleep_with_visual_lock(handle, self.waypoint_delay)

    def _move_so101_to_xyz(
        self,
        handle: mujoco.viewer.Handle,
        xyz: tuple[float, float, float],
        *,
        gripper: float,
        phase: str,
        steps: int,
        held_player: Player | None = None,
    ) -> None:
        with handle.lock():
            joint_targets = self.env.solve_so101_ik(xyz)
            trajectory = self.env.interpolate_so101_joint_trajectory(joint_targets, steps=steps)
        for index, waypoint in enumerate(trajectory):
            if not handle.is_running() or self.stopped:
                return
            while self.paused and handle.is_running() and not self.stopped:
                self._sync(handle, phase=f"{phase}_paused")
                time.sleep(1.0 / 30.0)
            with handle.lock():
                self.env.set_so101_joint_targets(waypoint, gripper=gripper)
                self.env.simulate(6)
                if held_player is not None:
                    self._show_held_stone_at_gripper("so101", held_player)
            self._sync(handle, phase=f"{phase} {index + 1}/{len(trajectory)}")
            self._sleep_with_visual_lock(handle, self.waypoint_delay)

    def _set_panda_gripper(
        self,
        handle: mujoco.viewer.Handle,
        opening: float,
        *,
        phase: str,
        held_player: Player | None = None,
    ) -> None:
        with handle.lock():
            self.env.set_panda_gripper(opening)
            self.env.simulate(24)
            if held_player is not None:
                self._show_held_stone_at_gripper("panda", held_player)
        self._sync(handle, phase=phase)
        self._sleep_with_visual_lock(handle, self.waypoint_delay)

    def _set_so101_gripper(
        self,
        handle: mujoco.viewer.Handle,
        opening: float,
        *,
        phase: str,
        held_player: Player | None = None,
    ) -> None:
        with handle.lock():
            self.env.set_so101_gripper(opening)
            self.env.simulate(24)
            if held_player is not None:
                self._show_held_stone_at_gripper("so101", held_player)
        self._sync(handle, phase=phase)
        self._sleep_with_visual_lock(handle, self.waypoint_delay)

    def _show_held_stone_at_gripper(self, robot_model: str, player: Player) -> None:
        if robot_model == "panda":
            x, y, z = self.env.panda_gripper_world()
            z -= 0.030
        else:
            x, y, z = self.env.so101_gripper_world()
            z -= 0.014
        if self.env.held_stone_player is None:
            self.env.grasp_supply_stone(player, x, y, z)
        else:
            self.env.set_held_stone_world(x, y, z, player)

    def _set_camera(self, handle: mujoco.viewer.Handle) -> None:
        camera_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        if camera_id < 0:
            raise ValueError(f"unknown camera: {self.camera}")
        with handle.lock():
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            handle.cam.fixedcamid = camera_id

    def _sync(self, handle: mujoco.viewer.Handle, phase: str | None = None) -> None:
        left, right = self.env.status_lines()
        phase_line = f"phase {phase}" if phase else self.last_status
        pause_line = "paused" if self.paused else "running"
        handle.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "space pause/resume\nn one move while paused\nq close",
                f"{pause_line}\n{left}\n{right}\n{phase_line}",
            )
        )
        self._lock_viewer_visuals(handle)
        handle.sync()
        self._lock_viewer_visuals(handle)

    def _handle_key(self, key: int) -> None:
        if key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_N:
            self.single_step_requested = True
        elif key in {glfw.KEY_Q, glfw.KEY_ESCAPE}:
            self.stopped = True
        if self._active_handle is not None:
            self._lock_viewer_visuals(self._active_handle)

    def _sleep_with_visual_lock(self, handle: mujoco.viewer.Handle, seconds: float) -> None:
        deadline = time.monotonic() + max(seconds, 0.0)
        while handle.is_running() and not self.stopped:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            self._lock_viewer_visuals(handle)
            time.sleep(min(remaining, 1.0 / 120.0))

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
    parser = argparse.ArgumentParser(description="Watch a trained Gomoku checkpoint play in the MuJoCo viewer.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--rule-set", choices=("free", "renju"))
    parser.add_argument("--center-opening", action="store_true")
    parser.add_argument("--no-center-opening", action="store_true")
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--camera", choices=("top", "iso", "robot_full"), default="robot_full")
    parser.add_argument("--move-delay", type=float, default=0.35)
    parser.add_argument("--waypoint-delay", type=float, default=0.18)
    parser.add_argument("--robot-model", choices=("kinematic", "panda", "so101"), default="so101")
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

    AIMujocoViewer(
        checkpoint,
        simulations=args.simulations,
        device=args.device,
        win_length=args.win_length,
        rule_set=args.rule_set,
        enforce_center_opening=enforce_center_opening,
        max_moves=args.max_moves,
        camera=args.camera,
        move_delay=args.move_delay,
        waypoint_delay=args.waypoint_delay,
        robot_model=args.robot_model,
    ).run()


if __name__ == "__main__":
    main()
