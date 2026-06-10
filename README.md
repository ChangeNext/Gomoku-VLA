# Gomoku-VLA

MuJoCo 기반 오목 보드게임 로봇 시스템을 위한 초기 MVP입니다.

## 현재 포함된 것

- 15x15 오목 board state, 턴 관리, 승리 판정
- MuJoCo XML을 동적으로 생성하는 오목판 시뮬레이션
- 사람 착수 테스트용 CLI
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

PNG snapshot을 만들려면:

```bash
python -m scripts.render_snapshot
```

## 테스트

```bash
python -m unittest discover -s tests
```
