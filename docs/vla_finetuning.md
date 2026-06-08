# VLA Fine-tuning

VLA 단계는 두 개로 분리한다.

## Coordinate Execution VLA

오목 전략은 외부 policy가 결정하고, VLA는 지정된 board 좌표에 돌을 놓는 조작만 수행한다.

## Gomoku-aware VLA

VLA가 board state와 목표를 함께 보고 다음 착수와 조작을 모두 생성한다.

Safety layer는 두 단계 모두에서 VLA 외부에 유지한다.
