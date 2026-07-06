# MuJoCo Environment

## MVP Scope

현재 환경은 오목판, 흑/백 돌, 로봇팔을 MuJoCo scene으로 표현하는 최소 실행 버전이다. 기본 경로는 기존 scripted kinematic arm을 유지하며, `robot_model="so101"`을 선택하면 MuJoCo Menagerie의 Robot Studio SO-101 MJCF를 scene에 merge한다. `robot_model="panda"`는 비교용으로 남겨둔 full-size Franka Panda 모델이다.

- board size: 15x15
- cell size: 3.5 cm
- black/white stones: fixed cylinder geoms
- human input: Matplotlib 클릭 기반 오목판
- robot view: MuJoCo 렌더 기반 Franka 전체 뷰
- board/world 좌표 변환 제공
- 착수 시 모델 재생성 없이 stone rgba, cursor, gripper 위치를 갱신
- 사람 착수는 gripper 위치를 유지하고, 로봇 착수만 gripper target을 갱신
- `top`, `iso`, `robot_full` camera 지원

## Scene Design

사용자가 실제 일상 공간에서 오목을 두는 느낌을 주기 위해 MuJoCo 장면에 다음 visual 요소를 포함한다.

- wooden table
- floor and room walls
- cup and notebook props
- raised Gomoku board
- scripted kinematic Franka-style arm with base, seven visual links, wrist, hand, and fingers
- optional Menagerie Robot Studio SO-101 with 5 arm joints, 1 gripper joint, 6 actuators, and `so101_ee_site` / `so101_gripper_site`
- optional Menagerie Franka Emika Panda with 7 arm joints, 2 finger joints, 8 actuators, and `panda_ee_site` / `panda_gripper_site`

Interactive UI는 왼쪽에 사람이 클릭해서 둘 수 있는 오목판을 두고, 오른쪽에 `robot_full` 카메라 렌더를 배치해 로봇 전체와 판 주변 맥락이 보이도록 한다.

## Run

```bash
python -m scripts.interactive_play
```

스냅샷과 XML을 생성하려면:

```bash
python -m scripts.render_snapshot
```

## Coordinate Convention

- board 좌표는 `(row, col)`이다.
- `(0, 0)`은 렌더 기준 좌상단이다.
- world 좌표는 board 중심을 `(0, 0)`으로 둔다.
- `board_to_world()`와 `world_to_board()`로 변환한다.
- SO-101/Panda 제어 단계에서는 `board_to_world(row, col)`을 end-effector target pose의 translation으로 쓰고, IK가 joint target으로 변환한다.

## SO-101 Integration

SO-101 is the preferred tabletop arm for the board-scale manipulation stage. Menagerie assets are vendored under `third_party/mujoco_menagerie/robotstudio_so101`.

- joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`
- actuators: 5 arm position actuators plus 1 gripper position actuator
- sites: `so101_ee_site` and `so101_gripper_site`

Current SO-101 control connects target cells to a hover pose above the board, solves IK for `so101_ee_site`, and sends interpolated joint targets through the arm actuators. The IK objective now includes a vertical end-effector axis term so the gripper approaches the target cell from above instead of only matching translation. In SO-101/Panda viewer modes the selection cursor uses a neutral fixed color and SO-101 disables the moving key-light shadow, which keeps WASD/arrow hover motion from looking like board color changes. The scene also uses a flat fixed skybox/haze color, and `viewer_play.py` re-applies stable viewer visualization flags on every sync so keyboard movement does not leave debug shading modes enabled. The gripper actuator is wired for open/close commands. Stone attach/detach and physical grasp quality remain later stages.

The SO-101 base is placed just outside the playable grid on the board's right edge. This keeps the robot visually off the Gomoku intersections while preserving enough reach for the current 15x15 board hover targets.

In `viewer_play.py`, pressing `space` / `enter` now runs an articulated place sequence for SO-101/Panda: move to hover, descend to a lower place pose, open/release the gripper while the stone is committed to the board state, then retreat to hover. This is still a visual/controller sequence; physical stone attach/detach from a supply tray remains a later stage.

## Panda Integration

Menagerie assets are vendored under `third_party/mujoco_menagerie/franka_emika_panda`. The first integration stage only guarantees that the Panda model loads in the Gomoku scene at a home keyframe and exposes stable control anchors:

- joints: `joint1` through `joint7`, plus `finger_joint1` and `finger_joint2`
- actuators: Menagerie Panda's 7 arm position actuators plus gripper tendon actuator
- sites: `panda_ee_site` and `panda_gripper_site`

Current Panda control connects target cells to a hover pose above the board, solves position-only IK for `panda_ee_site`, and sends interpolated joint targets through the 7 arm actuators. The gripper tendon actuator is wired for open/close commands. Stone attach/detach and physical grasp quality remain later stages.

The Menagerie Panda is kept at real robot scale, which makes it visually too large for the current tabletop Gomoku scene. It remains useful as a reference articulated industrial arm, but SO-101 should be used for the main tabletop pick/place path.

## Next Steps

- extend hover-only target motion into pick/place waypoints
- add stone attach/detach or mocap constraints
- collision group과 workspace limit 추가
