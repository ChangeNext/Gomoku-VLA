# Human-playable Environment

## Current Viewer MVP

```bash
python -m scripts.viewer_play
```

- 기본은 `iso` 시점과 `top` 시점의 passive viewer 두 개를 띄운다.
- `arrow` / `wasd`로 커서를 이동하고 `space` 또는 `enter`로 착수한다.
- scene XML 재컴파일 없이 돌, 커서, 그리퍼 위치만 갱신한다.

MuJoCo Python viewer는 현재 구조상 마우스 클릭 callback을 직접 노출하지 않기 때문에, 현재 human input은 keyboard cursor 방식으로 둔다.

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
