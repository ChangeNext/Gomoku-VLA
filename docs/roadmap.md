# Roadmap

## 1. MuJoCo Simulation MVP

- 15x15 오목판 MuJoCo scene 생성
- 착수 시 board state와 MuJoCo geoms 동기화
- headless render와 XML export 지원

## 2. Human-playable Gomoku Environment

- CLI 착수 MVP
- 이후 GUI 또는 web interface에서 좌표 선택 지원

## 3. Rule/Search-based Baseline

- 합법 수 필터링
- 간단 heuristic policy
- Minimax baseline

## 4. Learning-based Move Policy

- self-play episode logging
- black/white DQN 분리 학습

## 5. VLA-based Manipulation

- board 좌표를 로봇 action target으로 변환
- IK, trajectory, gripper control 추가

## 6. Gomoku-aware VLA

- board state 인식과 전략 판단 통합
- safety controller는 VLA 외부에 유지

## 7. Sim2Real

- 카메라 캘리브레이션
- 실제 board/stones 인식
- placement error 평가
