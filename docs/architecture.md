# Architecture

## Modules

- `board/`: board matrix, 좌표 검증, 턴 관리, 승리 판정
- `simulation/`: MuJoCo model 생성, stepping, rendering, board/world 좌표 변환
- `interface/`: 사람 입력과 GUI/web interface 예정
- `gomoku_ai/`: rule, minimax, DQN policy 예정
- `robot_control/`: IK, trajectory, gripper control 예정
- `vision/`: board/stones 인식 예정
- `vla/`: VLA inference와 fine-tuning 예정
- `data/`: episode logging 예정
- `evaluation/`: 승률, 착수 성공률, latency, placement error 예정

## Current MVP Flow

1. 사람이 row/col 좌표를 입력한다.
2. `board.GomokuBoard`가 합법 수와 승리 여부를 판정한다.
3. `simulation.GomokuMujocoEnv`가 board state로 MuJoCo XML을 재생성한다.
4. MuJoCo model/data를 stepping 또는 rendering한다.
