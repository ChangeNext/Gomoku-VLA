# Sim2Real

## Current Interfaces

The project now has first-pass modules for the required sim-to-real boundaries:

- `robot_control.safety`: validates supply availability, legal target cells, workspace bounds, and scripted action traces before robot execution.
- `simulation.GomokuMujocoEnv`: tracks black/white supply counts and the currently held stone. `grasp_supply_stone()` decrements inventory, and `commit_held_stone_to_cell()` releases the held stone into the board state.
- `vision.board_detector`: provides a calibrated top-down grid detector that samples each board intersection and classifies it as empty, black, or white.

These are still baseline interfaces. The grasp path uses scripted attachment for the carried stone, not contact-stable MuJoCo grasping. The vision detector assumes a calibrated top-down camera view; real camera use still needs calibration, lens correction, lighting checks, and confidence/error handling.

## Planned Work

- 실제 오목판 좌표계 캘리브레이션
- 카메라 기반 stone detection
- robot base와 board frame 정합
- placement error 측정
- joint/workspace/velocity/collision safety controller 유지
