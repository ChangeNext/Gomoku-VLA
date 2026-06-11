# Gomoku-VLA

MuJoCo 기반 오목 보드게임 로봇 시스템을 위한 초기 MVP입니다.

## 현재 포함된 것

- 15x15 오목 board state, 턴 관리, 승리 판정
- MuJoCo XML을 동적으로 생성하는 오목판 시뮬레이션
- 사람 착수 테스트용 CLI
- AlphaZero-style 9x9 self-play 학습 scaffold
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

학습된 9x9 checkpoint와 터미널에서 대국하려면:

```bash
python -m scripts.play_ai_cli --checkpoint checkpoints/alphazero_9x9_long.pt
```

AI가 흑으로 먼저 두게 하려면:

```bash
python -m scripts.play_ai_cli \
  --checkpoint checkpoints/alphazero_9x9_long.pt \
  --human white \
  --simulations 64
```

PNG snapshot을 만들려면:

```bash
python -m scripts.render_snapshot
```

## 9x9 AI 학습

학습 기능은 PyTorch가 필요하다. learning extra를 설치한 환경에서 실행한다.

```bash
pip install -e ".[learning]"
```

짧은 smoke training:

```bash
python -m scripts.train_alphazero \
  --board-size 9 \
  --win-length 5 \
  --iterations 5 \
  --games 20 \
  --simulations 64 \
  --epochs 4 \
  --batches-per-epoch 16 \
  --batch-size 256 \
  --device cuda \
  --checkpoint checkpoints/alphazero_9x9_mid.pt
```

기존 checkpoint에서 더 오래 이어 학습:

```bash
python -m scripts.train_alphazero \
  --board-size 9 \
  --win-length 5 \
  --iterations 20 \
  --games 40 \
  --simulations 128 \
  --epochs 8 \
  --batches-per-epoch 64 \
  --batch-size 256 \
  --device cuda \
  --initial-checkpoint checkpoints/alphazero_9x9_long.pt \
  --checkpoint checkpoints/alphazero_9x9_deep.pt
```

더 오래 돌릴 때는 `--iterations`, `--games`, `--simulations`, `--epochs`, `--batches-per-epoch`를 함께 늘린다. 현재 학습 업데이트 수는 다음과 같다.

```text
gradient steps = iterations * epochs * batches_per_epoch
```

학습 batch에는 회전/반사 augmentation이 기본 적용된다. 정확한 샘플 디버깅이 필요할 때만 `--no-augment`를 사용한다. `checkpoints/`는 git에 올라가지 않으므로 학습 결과는 로컬 파일로 관리한다.

## 테스트

```bash
python -m unittest discover -s tests
```
