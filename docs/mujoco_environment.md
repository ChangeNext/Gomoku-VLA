# MuJoCo Environment

## MVP Scope

현재 환경은 오목판과 돌을 MuJoCo scene으로 표현하는 최소 실행 가능 버전이다.

- board size: 15x15
- black/white stones: fixed cylinder geoms
- simple robot arm: visual geoms that point to the last move target
- board/world 좌표 변환 제공
- 착수 후 model 재생성 없이 geom rgba / geom pos만 갱신
- `top`, `iso` camera 지원

## Run

```bash
python -m scripts.render_snapshot
```

이 명령은 다음 파일을 생성한다.

- `gomoku_snapshot.ppm`: top camera render
- `gomoku_scene.xml`: 현재 MuJoCo XML

## Coordinate Convention

- board 좌표는 `(row, col)`이다.
- `(0, 0)`은 렌더링 기준 좌상단이다.
- world 좌표는 board 중심을 `(0, 0)`으로 둔다.
- `board_to_world()`와 `world_to_board()`로 변환한다.

## Next Steps

- fixed stone geoms를 실제 pick-and-place target으로 확장
- robot arm, gripper, stone tray 추가
- collision group과 workspace limit 추가
