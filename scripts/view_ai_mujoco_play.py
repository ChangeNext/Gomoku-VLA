from __future__ import annotations

import argparse
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer

from gomoku_ai.inference import CheckpointPolicy
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

    def run(self) -> None:
        handle = mujoco.viewer.launch_passive(
            self.env.model,
            self.env.data,
            key_callback=self._handle_key,
            show_left_ui=False,
            show_right_ui=False,
        )
        try:
            self._set_camera(handle)
            self._sync(handle)
            while handle.is_running() and not self.stopped:
                if self.paused and not self.single_step_requested:
                    self._sync(handle)
                    time.sleep(1.0 / 30.0)
                    continue
                if self.env.board.winner is not None or self.env.board.move_count >= self.max_moves:
                    self.last_status = "finished"
                    self._sync(handle)
                    time.sleep(1.0 / 30.0)
                    continue
                single_step = self.single_step_requested
                self.single_step_requested = False
                if single_step:
                    self.paused = False
                self._play_ai_move(handle)
                if single_step:
                    self.paused = True
        finally:
            handle.close()

    def _play_ai_move(self, handle: mujoco.viewer.Handle) -> None:
        player = self.env.board.current_player
        prediction = self.policy.predict(self.env.board)
        row, col = prediction.move
        self.last_status = (
            f"{player.name.lower()} -> ({row}, {col}) "
            f"value={prediction.value:.3f}"
        )
        with handle.lock():
            self.env.set_selection(row, col, update_robot_target=False)
        self._sync(handle)
        time.sleep(self.move_delay)

        if self.robot_model == "panda":
            self._play_panda_target_move(handle, row, col)
        elif self.robot_model == "so101":
            self._play_so101_target_move(handle, row, col)
        else:
            self._play_kinematic_pick_place(handle, row, col, player)

        with handle.lock():
            self.env.step((row, col), update_robot_target=True)
        self._sync(handle, phase="placed")
        time.sleep(self.move_delay)

    def _play_kinematic_pick_place(self, handle: mujoco.viewer.Handle, row: int, col: int, player) -> None:
        robot_action = build_pick_place_action(self.env, (row, col), player)
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
            self._sync(handle, phase=str(point["phase"]))
            time.sleep(self.waypoint_delay)

    def _play_panda_target_move(self, handle: mujoco.viewer.Handle, row: int, col: int) -> None:
        with handle.lock():
            target_xyz = self.env.panda_target_pose_for_cell(row, col)
            joint_targets = self.env.solve_panda_ik(target_xyz)
            trajectory = self.env.interpolate_panda_joint_trajectory(joint_targets, steps=90)
            self.env.set_panda_gripper(1.0)

        for index, waypoint in enumerate(trajectory):
            if not handle.is_running() or self.stopped:
                return
            while self.paused and handle.is_running() and not self.stopped:
                self._sync(handle, phase="panda_paused")
                time.sleep(1.0 / 30.0)
            with handle.lock():
                self.env.set_panda_joint_targets(waypoint, gripper=1.0)
                self.env.simulate(6)
            self._sync(handle, phase=f"panda_move {index + 1}/{len(trajectory)}")
            time.sleep(self.waypoint_delay)

        for gripper, phase in ((0.0, "gripper_close"), (1.0, "gripper_open")):
            if not handle.is_running() or self.stopped:
                return
            with handle.lock():
                self.env.set_panda_gripper(gripper)
                self.env.simulate(20)
            self._sync(handle, phase=phase)
            time.sleep(self.waypoint_delay)

    def _play_so101_target_move(self, handle: mujoco.viewer.Handle, row: int, col: int) -> None:
        with handle.lock():
            target_xyz = self.env.so101_target_pose_for_cell(row, col)
            joint_targets = self.env.solve_so101_ik(target_xyz)
            trajectory = self.env.interpolate_so101_joint_trajectory(joint_targets, steps=80)
            self.env.set_so101_gripper(1.0)

        for index, waypoint in enumerate(trajectory):
            if not handle.is_running() or self.stopped:
                return
            while self.paused and handle.is_running() and not self.stopped:
                self._sync(handle, phase="so101_paused")
                time.sleep(1.0 / 30.0)
            with handle.lock():
                self.env.set_so101_joint_targets(waypoint, gripper=1.0)
                self.env.simulate(6)
            self._sync(handle, phase=f"so101_move {index + 1}/{len(trajectory)}")
            time.sleep(self.waypoint_delay)

        for gripper, phase in ((0.0, "gripper_close"), (1.0, "gripper_open")):
            if not handle.is_running() or self.stopped:
                return
            with handle.lock():
                self.env.set_so101_gripper(gripper)
                self.env.simulate(20)
            self._sync(handle, phase=phase)
            time.sleep(self.waypoint_delay)

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
        handle.sync()

    def _handle_key(self, key: int) -> None:
        if key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_N:
            self.single_step_requested = True
        elif key in {glfw.KEY_Q, glfw.KEY_ESCAPE}:
            self.stopped = True


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
    parser.add_argument("--robot-model", choices=("kinematic", "panda", "so101"), default="kinematic")
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
