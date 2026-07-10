# Human-playable Environment

Project goal: human and viewer tools support a Gomoku-aware VLA pipeline. Human clicks and AI moves are useful for evaluation, but VLA training inputs must not include the answer coordinate.

## Current Viewer MVP

```bash
python -m scripts.viewer_play
```

- 기본은 `iso` 시점과 `top` 시점의 passive viewer 두 개를 띄운다.
- `arrow` / `wasd`로 커서를 이동하고 `space` 또는 `enter`로 착수한다.
- scene XML 재컴파일 없이 돌, 커서, 그리퍼 위치만 갱신한다.

MuJoCo Python viewer는 현재 구조상 마우스 클릭 callback을 직접 노출하지 않기 때문에, 현재 human input은 keyboard cursor 방식으로 둔다.

## Current Click UI MVP

```bash
python -m scripts.interactive_play
```

- 사용자는 시작 전에 뜨는 팝업 창에서 `Black` 또는 `White`를 눌러 자기 돌 색을 선택한다.
- 사람이 클릭해서 두는 착수는 board state와 돌 geom만 갱신하고, 로봇팔 target은 갱신하지 않는다.
- 로봇 차례에는 현재 baseline policy가 중앙에 가까운 합법 수를 선택하며, 이때만 로봇팔 target을 갱신한다.
- 게임 종료 시 사용자가 선택한 돌 기준으로 `승리`, `패배`, `무승부` 결과 팝업을 표시한다.
- `Reset`은 선택한 색을 유지한 채 새 게임을 시작한다. 사용자가 `White`이면 로봇이 흑 첫 수를 둔다.

## CLI MVP

```bash
python -m scripts.play_cli
```

입력 형식은 `row col`이다.

예:

```text
7 7
```

## Planned UI

- MuJoCo viewer 또는 web canvas에서 교차점 클릭
- invalid move rejection
- current player 표시
- winner/draw 상태 표시
- replay 저장
