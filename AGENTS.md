# AGENTS.md

## Project

이 프로젝트는 **Gomoku-VLA**이다.

목표는 MuJoCo 기반 시뮬레이션에서 사람이 직접 오목을 둘 수 있는 환경을 만들고, 로봇팔이 오목판 상태를 인식해 다음 수를 두는 **VLA 기반 보드게임 로봇 시스템**으로 확장하는 것이다.

최종 목표는 단순 pick-and-place가 아니라, 오목 전략 판단과 로봇 조작을 통합한 **Gomoku-aware VLA**이다.

## Development Policy

- 처음부터 End-to-End VLA로 구현하지 않는다.
- 먼저 MuJoCo 기반 simulation MVP를 만든다.
- 사람이 직접 착수할 수 있는 human-playable 환경을 반드시 고려한다.
- 초기 baseline은 rule/search-based 방식으로 구현한다.
- Minimax는 search-based로, DQN은 learning-based 또는 RL-based로 구분한다.
- 2개의 DQN을 사용하여 서로가 적대적으로 학습할 수 있도록 한다.
- VLA가 좌표 실행만 하는 단계와 오목 전략까지 판단하는 단계를 구분한다.
- safety layer는 VLA 밖에 별도로 유지한다.
- 새 기능을 추가하면 관련 `docs/` 문서도 함께 업데이트한다.
- context7을 활용하여 라이브러리 효율적으로 사용한다.

## System Direction

권장 개발 순서:

1. MuJoCo Simulation MVP
2. Human-playable Gomoku Environment
3. Rule/Search-based Baseline
4. Learning-based Move Policy
5. VLA-based Manipulation
6. Gomoku-aware VLA
7. Sim2Real

## Module Boundaries

가능하면 다음 모듈을 분리한다.

- `simulation/`: MuJoCo 환경, 로봇, 오목판, 물리 설정
- `interface/`: 사람 입력, GUI, 웹 인터페이스
- `vision/`: 오목판 및 돌 인식
- `board/`: board matrix, 좌표 변환, 턴 관리
- `gomoku_ai/`: Minimax, DQN, move policy
- `robot_control/`: IK, trajectory, gripper control
- `vla/`: VLA inference, fine-tuning, action chunking
- `data/`: 데이터 수집, episode logging
- `evaluation/`: 승률, 착수 성공률, latency, placement error 평가

## Documentation

상세 내용은 `docs/` 아래에 작성한다.

- `docs/roadmap.md`
- `docs/architecture.md`
- `docs/mujoco_environment.md`
- `docs/human_playable_environment.md`
- `docs/data_collection.md`
- `docs/vla_finetuning.md`
- `docs/sim2real.md`
- `docs/evaluation.md`
- `docs/study.md` # 자기소개서 및 면접을 위한 해당 프로젝트 이론 기술 기술

## Safety

로봇 action을 만들 때는 항상 다음 조건을 고려한다.

- joint limit
- workspace limit
- velocity limit
- collision check
- invalid move rejection
- emergency stop

VLA policy를 사용하더라도 safety controller는 제거하지 않는다.