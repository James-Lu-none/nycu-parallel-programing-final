from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


TARGET_NAMES = {"PhysicsStep", "RenderStep", "accelerations"}


def extract_target_means(csv_path: Path) -> Dict[str, str]:
    """
    Pull PhysicsStep and RenderStep mean_ns values from a Tracy CSV export.
    Returns a mapping of target name -> mean_ns (as a string from the CSV).
    Missing entries are omitted.
    """
    found: Dict[str, str] = {}

    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            name = row.get("name")
            if name in TARGET_NAMES:
                mean_value = row.get("mean_ns", "")
                found[name] = mean_value

            if len(found) == len(TARGET_NAMES):
                break

    return found


def write_summary(summary_path: Path, rows: Dict[str, Dict[str, str]]) -> None:
    """Write the summary file with a CSV-style header for easy parsing."""
    lines = ["name,PhysicsStep_mean_ns,RenderStep_mean_ns"]
    for filename, values in sorted(rows.items()):
        physics = values.get("PhysicsStep", "")
        render = values.get("RenderStep", "")
        lines.append(f"{filename},{physics},{render}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    eval_dir = Path("eval")
    csv_files = sorted(eval_dir.glob("*.csv"))

    results: Dict[str, Dict[str, str]] = {}
    for csv_file in csv_files:
        results[csv_file.name] = extract_target_means(csv_file)

    summary_path = eval_dir / "summary.csv"
    write_summary(summary_path, results)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
