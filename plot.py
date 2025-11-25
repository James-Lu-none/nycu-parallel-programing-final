from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

COLUMN_NAMES = ["name", "PhysicsStep_mean_ns", "RenderStep_mean_ns"]


def find_summary() -> Path:
    """Use summary.csv in CWD; fall back to eval/summary.csv if not present."""
    primary = Path("summary.csv")
    if primary.exists():
        return primary

    fallback = Path("eval") / "summary.csv"
    if fallback.exists():
        return fallback

    raise FileNotFoundError("summary.csv not found in current directory or eval/")


def read_summary(path: Path) -> Tuple[List[str], List[float], List[float]]:
    """
    Load summary.csv and return (names, physics_ms, render_ms).
    For names starting with "render", PhysicsStep is omitted (NaN).
    For names starting with "accelerations", RenderStep is omitted (NaN).
    """
    names: List[str] = []
    physics_ms: List[float] = []
    render_ms: List[float] = []

    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            name = row[COLUMN_NAMES[0]]
            physics_ns = row.get(COLUMN_NAMES[1], "") or "nan"
            render_ns = row.get(COLUMN_NAMES[2], "") or "nan"

            names.append(name)

            if name.startswith("render"):
                physics_ms.append(np.nan)
            else:
                physics_ms.append(float(physics_ns) / 1_000_000)

            if name.startswith("accelerations"):
                render_ms.append(np.nan)
            else:
                render_ms.append(float(render_ns) / 1_000_000)

    return names, physics_ms, render_ms


def plot_lines(names: List[str], physics_ms: List[float], render_ms: List[float]) -> None:
    x = np.arange(len(names))

    plt.figure(figsize=(14, 8))
    plt.plot(x, physics_ms, marker="o", label="PhysicsStep (ms)")
    plt.plot(x, render_ms, marker="o", label="RenderStep (ms)")

    plt.xticks(x, names, rotation=75, ha="right")
    plt.ylabel("mean time (ms)")
    plt.xlabel("csv file")
    plt.title("PhysicsStep / RenderStep mean time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    summary_path = find_summary()
    output_path = Path("plot.png")

    names, physics_ms, render_ms = read_summary(summary_path)
    plot_lines(names, physics_ms, render_ms)
    plt.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
