# Architecture

## Modules

- `board/`: board matrix, 좌표 검증, 턴 관리, 자유룰/렌주 승리와 금수 판정
- `simulation/`: MuJoCo model 생성, stepping, rendering, board/world 좌표 변환
- `interface/`: 사람 입력과 GUI/web interface 예정
- `gomoku_ai/`: AlphaZero-style policy/value model, MCTS, self-play data generation
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

## Learning Policy Flow

1. `gomoku_ai.encoding`이 `GomokuBoard`를 현재 플레이어 관점 tensor로 변환한다.
2. policy/value model이 착수 prior와 현재 판세 value를 예측한다.
3. `gomoku_ai.mcts`가 legal move만 대상으로 PUCT 탐색을 수행한다. 렌주 학습에서는 흑 중앙 첫 수, 3-3/4-4/장목 금수가 이 legal move mask에 반영된다.
4. self-play는 `(state, MCTS policy, final result)` sample을 저장한다.
5. replay buffer에서 mini-batch를 뽑아 policy loss와 value loss를 AdamW로 함께 최적화한다.
6. evaluator match가 기준을 넘으면 `best.pt`로 승격한다.
7. 학습된 policy가 선택한 `(row, col)`은 이후 `simulation.GomokuMujocoEnv.step()` 또는 로봇 제어 모듈로 전달한다.

카메라 인식과 로봇팔 제어는 AlphaZero 전략 학습과 분리한다. 최종 통합에서는 `camera image -> board state -> policy -> row,col -> robot action` 순서로 연결한다.
