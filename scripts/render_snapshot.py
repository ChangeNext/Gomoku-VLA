from __future__ import annotations

from pathlib import Path

from PIL import Image

from simulation import GomokuMujocoEnv


def write_png(path: Path, pixels: object) -> None:
    Image.fromarray(pixels).save(path)


def main() -> None:
    env = GomokuMujocoEnv()
    for move in [(7, 7), (7, 8), (8, 8), (6, 6), (8, 7)]:
        env.step(move)
    env.simulate(5)
    output = Path("gomoku_snapshot.png")
    write_png(output, env.render(width=900, height=900))
    env.export_model("gomoku_scene.xml")
    print(f"Wrote {output} and gomoku_scene.xml")


if __name__ == "__main__":
    main()
