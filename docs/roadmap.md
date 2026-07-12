# Roadmap and Current Status

The project aims to observe a Gomoku board, choose a strong legal move, and
safely execute it. Coordinate-following placement is an auxiliary execution
skill, not the strategic objective.

## Implemented Foundation

- Free-rule and Renju board state with legal moves, wins, draws, center opening,
  and Black overline/double-three/double-four restrictions
- Dynamic MuJoCo Gomoku scene with board/world coordinate conversion, multiple
  cameras, stone state, and headless rendering
- CLI, Matplotlib click UI, keyboard MuJoCo viewer, and AI viewer
- Kinematic baseline plus vendored articulated SO-101 and Panda integrations
- AlphaZero-style residual policy/value network, batched MCTS, self-play,
  replay buffer, augmentation, training outputs, and checkpoint evaluation
- Shared checkpoint inference with tactical immediate-win/threat handling
- Policy episode recording and MuJoCo SO-101 multi-view collection
- OpenVLA-OFT-style custom manifest export with input/target separation
- Workspace, inventory, legality, and scripted trajectory safety checks
- Calibrated-grid brightness/contrast vision baseline
- Local browser human-checkpoint evaluation and persistent results

Implemented means that code and regression coverage exist. It does not imply
strong 15x15 play, production robot safety, or sim-to-real readiness.

## Near-term Priorities

### 1. Establish a strong strategy teacher

- run and compare reproducible 15x15 Renju training runs;
- expand evaluator games and fixed tactical position suites;
- quantify illegal rate, color-split win rate, and checkpoint variance;
- improve Renju edge-case coverage before treating it as competition-complete.

### 2. Validate perception as a policy input

- add automatic/robust board calibration and confidence estimates;
- test perspective, lighting, occlusion, and camera noise;
- validate observed transitions against the previous board state;
- measure board-cell classification and full-board reconstruction accuracy.

### 3. Harden SO-101 data generation

- quantify grasp, carry, placement, and final-cell success;
- add joint, velocity, collision, and timeout checks;
- replace or compare constraint attachment with contact-stable grasping;
- add retries or closed-loop corrections without hiding failed attempts.

### 4. Train the Gomoku-aware VLA

- integrate the exported multi-view manifest with an actual OpenVLA/OFT trainer;
- first validate move-token prediction from board observations;
- then validate action prediction conditioned on the model's move token;
- finally train and evaluate the combined `<MOVE_k> <ACT_...> <EOS>` output;
- keep coordinate-leaking prompts out of strategic training splits.

### 5. End-to-end and sim-to-real evaluation

- report perception, strategy, safety, and manipulation metrics separately;
- add real camera, board-frame, and robot-base calibration;
- integrate real emergency stop, watchdog, collision, and joint-limit controls;
- measure real placement error and recovery behavior;
- compare simulation and real-world distributions and failures.

## Completion Gates

The project should not be described as an end-to-end Gomoku-aware VLA until it
can demonstrate all of the following on held-out games:

- board state inferred from images without answer-coordinate leakage;
- selected move is legal and strategically competitive;
- safety gate authorizes or rejects execution with an auditable reason;
- robot completes the selected move within defined placement tolerance;
- failures are attributed to perception, strategy, safety, or execution;
- real-robot claims are backed by real-robot tests, not MuJoCo results alone.
