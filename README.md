# Gomoku-VLA

Project goal: build a Gomoku-aware VLA that reads the board, chooses a strong legal move, and then executes that move. The dataset must avoid leaking the target row/column into model inputs; selected moves, target coordinates, and robot trajectories are supervision labels or metadata, not instructions.

MuJoCo 기반 오목 보드게임 로봇 시스템을 위한 초기 MVP입니다.

## 문서

프로젝트 구조, 게임 규칙, 학습, 시뮬레이션, 데이터, 안전, 비전 및
평가 문서는 [`docs/index.md`](docs/index.md)에서 주제별로 찾을 수 있습니다.
에이전트 작업 규칙과 저장소 지도는 [`AGENTS.md`](AGENTS.md)를 따릅니다.

## 현재 포함된 것

- 15x15 오목/렌주 board state, 턴 관리, 승리/금수 판정
- MuJoCo XML을 동적으로 생성하는 오목판 시뮬레이션
- 사람 착수 테스트용 CLI
- AlphaZero-style 15x15 렌주 self-play 학습 scaffold
- 학습 checkpoint와 대국하는 터미널 AI CLI
- MuJoCo 렌더링 snapshot 생성 스크립트

## 실행

팝업에서 흑/백을 선택하고 클릭 UI로 플레이하려면:

```bash
python -m scripts.interactive_play
```

사람이 클릭해서 둔 수는 로봇팔 target을 갱신하지 않고, 로봇 차례의 수만 로봇팔 target을 갱신한다.

MuJoCo viewer 중심으로 보려면:

```bash
python -m scripts.viewer_play
```

기본은 `iso` + `top` 두 개의 viewer 창을 띄운다. 한 개만 띄우려면:

```bash
python -m scripts.viewer_play --single-view
```

조작:

- `arrow` / `wasd`: 커서 이동
- `space` / `enter`: 착수
- `r`: 리셋

터미널에서 좌표를 입력하려면:

```bash
python -m scripts.play_cli
```

학습된 checkpoint와 터미널에서 대국하려면:

```bash
python -m scripts.play_ai_cli --checkpoint gomoku_ai/runs/<run-name>/checkpoints/latest.pt
```

AI가 흑으로 먼저 두게 하려면:

```bash
python -m scripts.play_ai_cli \
  --checkpoint gomoku_ai/runs/2026-06-23_15x15_renju_resnet/checkpoints/latest.pt \
  --human white \
  --simulations 64
```

PNG snapshot을 만들려면:

```bash
python -m scripts.render_snapshot
```

## AI 학습

학습 기능은 PyTorch가 필요하다. learning extra를 설치한 환경에서 실행한다.

```bash
pip install -e ".[learning]"
```

학습 중에는 `tqdm`으로 전체 iteration, self-play game, optimization batch, evaluator game 진행률과 ETA를 표시한다. 로그 파일처럼 progress bar가 필요 없는 실행에서는 `--no-progress`를 사용한다.

짧은 smoke training:

```bash
python -m scripts.train_alphazero \
  --board-size 5 \
  --win-length 4 \
  --rule-set free \
  --no-center-opening \
  --iterations 1 \
  --games 2 \
  --simulations 8 \
  --epochs 1 \
  --batches-per-epoch 1 \
  --batch-size 32 \
  --run-name smoke_free_5x5
```

학습 산출물은 `gomoku_ai/runs/<run-name>/` 아래에 한 번에 저장된다.

```text
checkpoints/latest.pt
metrics/history.csv
plots/training.png
plots/policy_heatmap_empty.png
replay/replay_buffer.pkl
config.json
```

기본 학습 방향은 15x15 렌주다. 흑 첫 수는 중앙으로 강제하고, 흑 3-3/4-4/장목 금수를 legal move에서 제외한다. self-play 착수는 MCTS visit policy에서만 샘플링하며, 전술 보정은 사람 대국과 평가 move selection에만 사용한다.

15x15 렌주 장기 학습:

```bash
python -m scripts.train_alphazero \
  --iterations 80 \
  --games 80 \
  --simulations 256 \
  --epochs 8 \
  --batches-per-epoch 128 \
  --batch-size 512 \
  --replay-capacity 500000 \
  --learning-rate 3e-4 \
  --device cuda \
  --amp \
  --run-name 2026-06-23_15x15_renju_resnet \
  --channels 256 \
  --res-blocks 16 \
  --input-channels 6 \
  --evaluation-games 20 \
  --evaluation-simulations 64
```

더 오래 돌릴 때는 `--iterations`, `--games`, `--simulations`, `--epochs`, `--batches-per-epoch`를 함께 늘린다. 현재 학습 업데이트 수는 다음과 같다.

```text
gradient steps = iterations * epochs * batches_per_epoch
```

학습 batch에는 회전/반사 augmentation이 기본 적용된다. 정확한 샘플 디버깅이 필요할 때만 `--no-augment`를 사용한다. 기본 모델은 AlphaZero-style ResNet CNN policy/value network이고 optimizer는 AdamW다. CUDA에서 `--amp`를 켜면 mixed precision 학습을 사용한다. `gomoku_ai/runs/`는 git에 올라가지 않으므로 학습 결과는 로컬 파일로 관리한다.

학습 중 `--history-csv`는 iteration별 loss, replay size, sample 수를 기록하고, `--plot`은 같은 내용을 PNG로 계속 갱신한다. 기존 CSV로 plot만 다시 만들려면 다음을 실행한다.

```bash
python -m scripts.plot_training_history gomoku_ai/runs/2026-06-23_15x15_renju_resnet/metrics/history.csv \
  --output gomoku_ai/runs/2026-06-23_15x15_renju_resnet/plots/training.png
```

checkpoint끼리 평가하려면:

```bash
python -m scripts.evaluate_checkpoint \
  --candidate gomoku_ai/runs/new_run/checkpoints/latest.pt \
  --baseline gomoku_ai/runs/old_run/checkpoints/latest.pt \
  --rule-set renju \
  --center-opening \
  --output-csv gomoku_ai/runs/new_run/metrics/evaluation.csv \
  --promote-to gomoku_ai/runs/new_run/checkpoints/best.pt
```

## 15x15 AI 학습

15x15는 action 수가 225개이고 한 판이 길어서 작은 보드보다 훨씬 느리다. 먼저 안정적인 장기 run을 만들고 `metrics/history.csv`, evaluator 점수, 직접 대국으로 확인한다.

```bash
python -m scripts.train_alphazero \
  --board-size 15 \
  --win-length 5 \
  --iterations 80 \
  --games 80 \
  --simulations 256 \
  --epochs 8 \
  --batches-per-epoch 128 \
  --batch-size 512 \
  --replay-capacity 500000 \
  --learning-rate 3e-4 \
  --device cuda \
  --amp \
  --run-name 2026-06-24_15x15_renju_resnet \
  --channels 256 \
  --res-blocks 16 \
  --input-channels 6 \
  --evaluation-games 20 \
  --evaluation-simulations 64
```

진행 확인:

```bash
tail -n 10 gomoku_ai/runs/2026-06-23_15x15_renju_resnet/metrics/history.csv
tmux attach -t gomoku_train_15x15
```

학습 중 정책 분포는 다음 이미지로 확인할 수 있다.

```bash
ls -lh gomoku_ai/runs/2026-06-23_15x15_renju_resnet/plots/policy_heatmap_empty.png
```

직접 테스트:

```bash
python -m scripts.play_ai_cli \
  --checkpoint gomoku_ai/runs/2026-06-23_15x15_renju_resnet/checkpoints/latest.pt \
  --human black \
  --simulations 64
```

AI 선공:

```bash
python -m scripts.play_ai_cli \
  --checkpoint gomoku_ai/runs/2026-06-23_15x15_renju_resnet/checkpoints/latest.pt \
  --human white \
  --simulations 64
```

## 테스트

```bash
python -m unittest discover -s tests
```
