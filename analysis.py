from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import pandas as pd


# Configure metrics to extract from Tracy CSV files
# Key: column name in summary CSV, Value: name field in Tracy CSV
METRICS_CONFIG = {
    "physicsstep_mean_ns": "PhysicsStep",
    "renderstep_mean_ns": "RenderStep",
    "accelerations_mean_ns": "accelerations",
}

# Configure which metrics to skip for files matching certain patterns
# Key: filename prefix, Value: set of metric column names to skip (set to NaN)
SKIP_RULES = {
    "render": {"physicsstep_mean_ns"},
    "accelerations": {"renderstep_mean_ns"},
}


def extract_metrics(csv_path: Path) -> Dict[str, str]:
    """
    Extract configured metrics from a Tracy CSV export.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary mapping metric column names to mean_ns values
    """
    found: Dict[str, str] = {}
    target_names = set(METRICS_CONFIG.values())
    
    try:
        with csv_path.open(newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                name = row.get("name", "")
                if name in target_names:
                    # Find which metric this corresponds to
                    for metric_col, metric_name in METRICS_CONFIG.items():
                        if name == metric_name:
                            mean_value = row.get("mean_ns", "")
                            found[metric_col] = mean_value
                            break
                
                # Early exit if we found all metrics
                if len(found) == len(target_names):
                    break
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    
    return found


def generate_summary(eval_dir: Path) -> pd.DataFrame:
    """
    Process all CSV files in eval directory and create summary DataFrame.
    
    Args:
        eval_dir: Directory containing Tracy CSV exports
        
    Returns:
        DataFrame with metrics for each CSV file
    """
    csv_files = sorted([f for f in eval_dir.glob("*.csv") 
                       if f.name != "summary.csv"])
    
    rows = []
    for csv_file in csv_files:
        metrics = extract_metrics(csv_file)
        row = {"csv_file_name": csv_file.name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Apply skip rules: set metrics to NaN for files matching patterns
    for prefix, skip_cols in SKIP_RULES.items():
        mask = df["csv_file_name"].str.startswith(prefix)
        for col in skip_cols:
            if col in df.columns:
                df.loc[mask, col] = pd.NA
    
    return df


def save_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Save summary DataFrame to CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Summary written to {output_path}")
    print(f"Processed {len(df)} CSV files")


def load_summary(summary_path: Path | None = None) -> pd.DataFrame:
    """
    Load summary CSV, trying multiple locations if path not specified.
    
    Args:
        summary_path: Optional explicit path to summary.csv
        
    Returns:
        DataFrame with summary data
    """
    if summary_path is None:
        # Try current directory first, then eval/
        if Path("summary.csv").exists():
            summary_path = Path("summary.csv")
        elif Path("eval/summary.csv").exists():
            summary_path = Path("eval/summary.csv")
        else:
            raise FileNotFoundError(
                "summary.csv not found in current directory or eval/"
            )
    
    return pd.read_csv(summary_path)


def plot_metrics(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create line plot visualization of all metrics.
    
    Args:
        df: DataFrame with metrics data
        output_path: Path to save plot image
    """
    # Convert nanoseconds to milliseconds for all metric columns
    metric_cols = [col for col in df.columns if col != "csv_file_name"]
    df_plot = df.copy()
    
    for col in metric_cols:
        df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce") / 1_000_000
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = range(len(df_plot))
    for col in metric_cols:
        # Create readable label from column name
        label = col.replace("_mean_ns", "").replace("_", " ").title()
        ax.plot(x, df_plot[col], marker="o", label=label)
    
    # Configure plot
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["csv_file_name"], rotation=75, ha="right")
    ax.set_ylabel("Mean Time (ms)")
    ax.set_xlabel("CSV File")
    ax.set_title("Tracy Performance Metrics")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.show()


def main() -> None:
    """Main execution function."""
    eval_dir = Path("eval")
    
    # Generate summary
    print("Extracting metrics from CSV files...")
    df = generate_summary(eval_dir)
    
    # Save summary
    summary_path = eval_dir / "summary.csv"
    save_summary(df, summary_path)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.to_string(index=False))
    
    # Generate plot
    print("\nGenerating visualization...")
    plot_path = Path("plot.png")
    plot_metrics(df, plot_path)


if __name__ == "__main__":
    main()