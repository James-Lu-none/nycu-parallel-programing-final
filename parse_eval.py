import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def load_trace(csv_path: str) -> pd.DataFrame:
    columns: Iterable[str] = (
        "name",
        "src_file",
        "src_line",
        "ns_since_start",
        "exec_time_ns",
    )
    df = pd.read_csv(
        csv_path,
        usecols=columns
    )
    df = df.dropna(subset=["exec_time_ns", "ns_since_start", "src_line", "src_file", "name"])
    df[["ns_since_start", "exec_time_ns"]] = df[["ns_since_start", "exec_time_ns"]].astype(
        "int64"
    )
    return df


def summarize(df: pd.DataFrame, start_ns: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "name",
                "src_file",
                "src_line",
                "total_ns",
                "total_perc",
                "counts",
                "mean_ns",
                "min_ns",
                "max_ns",
                "std_ns",
            ]
        )

    duration = max(df["ns_since_start"].max() - start_ns, 1)
    grouped = df.groupby("name", sort=False)
    summary = grouped.agg(
        src_file=("src_file", "first"),
        src_line=("src_line", "first"),
        total_ns=("exec_time_ns", "sum"),
        counts=("exec_time_ns", "size"),
        mean_ns=("exec_time_ns", "mean"),
        min_ns=("exec_time_ns", "min"),
        max_ns=("exec_time_ns", "max"),
        std_ns=("exec_time_ns", "std"),
    ).reset_index()
    summary["total_perc"] = summary["total_ns"] / duration * 100
    return summary[
        [
            "name",
            "src_file",
            "src_line",
            "total_ns",
            "total_perc",
            "counts",
            "mean_ns",
            "min_ns",
            "max_ns",
            "std_ns",
        ]
    ]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate all Tracy CSV exports in a directory and compute timing "
            "statistics after the fourth PhysicsStep ns_since_start."
        )
    )
    parser.add_argument("input_dir", help="Directory containing Tracy CSV exports")
    parser.add_argument("output_dir", help="Directory to write summary CSV files")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        print(f"No CSV files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    for csv_path in csv_files:
        df = load_trace(csv_path)
        physics_steps = df.loc[df["name"] == "PhysicsStep", "ns_since_start"]
        start_ns = int(physics_steps.iloc[3]) if len(physics_steps) >= 4 else 0
        summary = summarize(df[df["ns_since_start"] >= start_ns], start_ns)
        for _, row in summary.iterrows():
            print(f"{csv_path.name} - {row['name']}: {row['counts']} rows", file=sys.stderr)

        summary.to_csv(output_dir / csv_path.name, index=False)


if __name__ == "__main__":
    main()
