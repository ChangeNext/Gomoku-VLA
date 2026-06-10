# MuJoCo Environment

## MVP Scope

현재 환경은 오목판, 흑/백 돌, Franka Panda 스타일 로봇팔을 MuJoCo scene으로 표현하는 최소 실행 버전이다.

- board size: 15x15
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
- fuller Franka Panda-style arm with base, seven visual links, wrist, hand, and fingers

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

## Next Steps

- fixed stone geoms를 실제 pick-and-place target으로 확장
- robot arm, gripper, stone tray의 물리 제약 추가
- collision group과 workspace limit 추가
