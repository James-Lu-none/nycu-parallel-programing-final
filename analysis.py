#!/usr/bin/env python3
"""
Unified script to extract metrics from Tracy CSV exports, generate summary,
and create visualizations grouped by metric type and parallel variations.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

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
    "render": {"physicsstep_mean_ns", "accelerations_mean_ns"},
    "accelerations": {"renderstep_mean_ns"},
}

# Parallel variation configurations
ACCELERATION_VARIATIONS = [
    "pthread_blocked",
    "pthread_interleaved",
    "pthread_mutex_blocked",
    "pthread_mutex_interleaved",
    "pthread_mutex_simd_blocked",
    "pthread_mutex_simd_interleaved",
    "pthread_simd_blocked",
    "pthread_simd_interleaved",
    "serial_simd",
    "serial",
    "cuda_blocked",
    "cuda_interleaved",
]

RENDER_VARIATIONS = [
    "serial",
    "serial_simd",
    "pthread",
    "pthread_simd",
    "pthread_mutex",
    "pthread_mutex_simd",
    "cuda",
]


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


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse filename to extract type, variation, and thread count.
    
    Args:
        filename: CSV filename (e.g., "accelerations_pthread_blocked_04.csv")
        
    Returns:
        Tuple of (type, variation, thread_count)
        e.g., ("accelerations", "pthread_blocked", "04")
    """
    # Remove .csv extension
    name = filename.replace(".csv", "")
    
    # Match pattern: type_variation_threadcount
    # Thread count is optional (2 digits at the end)
    match = re.match(r"(accelerations|render)_(.+?)(?:_(\d{2}))?$", name)
    
    if match:
        file_type = match.group(1)
        variation = match.group(2)
        thread_count = match.group(3) if match.group(3) else None
        return file_type, variation, thread_count
    
    return "", "", ""


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
        file_type, variation, thread_count = parse_filename(csv_file.name)
        
        row = {
            "csv_file_name": csv_file.name,
            "type": file_type,
            "variation": variation,
            "thread_count": thread_count
        }
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


def plot_metric_overview(df: pd.DataFrame, metric_col: str, results_dir: Path) -> None:
    """
    Create overview plots for a single metric:
    1. All variations grouped by SIMD/non-SIMD (excluding CUDA)
    2. All variations as separate lines
    
    Args:
        df: DataFrame with metrics data
        metric_col: Name of the metric column
        results_dir: Directory to save plot image
    """
    # Filter out rows where this metric is NaN
    df_metric = df[df[metric_col].notna()].copy()
    
    if len(df_metric) == 0:
        return
    
    # Create subdirectory for this metric
    metric_name = metric_col.replace("_mean_ns", "")
    metric_dir = results_dir / metric_name
    metric_dir.mkdir(exist_ok=True)
    
    # Convert from ns to ms
    df_metric[f"{metric_col}_ms"] = pd.to_numeric(df_metric[metric_col], errors="coerce") / 1_000_000
    
    # Add thread number for sorting
    df_metric["thread_num"] = df_metric["thread_count"].apply(
        lambda x: int(x) if x and x.isdigit() else 0
    )
    
    # Classify variations as SIMD or non-SIMD
    df_metric["is_simd"] = df_metric["variation"].str.contains("simd", case=False)
    df_metric["is_cuda"] = df_metric["variation"].str.contains("cuda", case=False)
    
    # Get unique variations
    variations = df_metric["variation"].unique()
    
    # Plot 1: All variations (one line per variation)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for variation in sorted(variations):
        df_var = df_metric[df_metric["variation"] == variation].copy()
        df_var = df_var.sort_values("thread_num")
        
        # Create x-axis values based on thread count
        x_values = []
        for _, row in df_var.iterrows():
            if row["thread_count"]:
                x_values.append(int(row["thread_count"]))
            else:
                x_values.append(0)
        
        label = variation.replace("_", " ").title()
        ax.plot(x_values, df_var[f"{metric_col}_ms"], marker="o", 
                linewidth=2, markersize=6, label=label)
    
    metric_display = metric_name.replace("_", " ").title()
    ax.set_ylabel("Mean Time (ms)", fontsize=12)
    ax.set_xlabel("Thread Count", fontsize=12)
    ax.set_title(f"{metric_display} - All Variations", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    output_path = metric_dir / f"{metric_name}_overview_all.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Plot 2: SIMD vs non-SIMD (excluding CUDA)
    df_no_cuda = df_metric[~df_metric["is_cuda"]].copy()
    
    if len(df_no_cuda) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Non-SIMD variations
        df_non_simd = df_no_cuda[~df_no_cuda["is_simd"]]
        for variation in sorted(df_non_simd["variation"].unique()):
            df_var = df_non_simd[df_non_simd["variation"] == variation].copy()
            df_var = df_var.sort_values("thread_num")
            
            x_values = [int(row["thread_count"]) if row["thread_count"] else 0 
                       for _, row in df_var.iterrows()]
            
            label = variation.replace("_", " ").title()
            ax1.plot(x_values, df_var[f"{metric_col}_ms"], marker="o", 
                    linewidth=2, markersize=6, label=label)
        
        ax1.set_ylabel("Mean Time (ms)", fontsize=12)
        ax1.set_xlabel("Thread Count", fontsize=12)
        ax1.set_title(f"Non-SIMD Variations", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(fontsize=9)
        
        # SIMD variations
        df_simd = df_no_cuda[df_no_cuda["is_simd"]]
        for variation in sorted(df_simd["variation"].unique()):
            df_var = df_simd[df_simd["variation"] == variation].copy()
            df_var = df_var.sort_values("thread_num")
            
            x_values = [int(row["thread_count"]) if row["thread_count"] else 0 
                       for _, row in df_var.iterrows()]
            
            label = variation.replace("_", " ").title()
            ax2.plot(x_values, df_var[f"{metric_col}_ms"], marker="o", 
                    linewidth=2, markersize=6, label=label)
        
        ax2.set_ylabel("Mean Time (ms)", fontsize=12)
        ax2.set_xlabel("Thread Count", fontsize=12)
        ax2.set_title(f"SIMD Variations", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(fontsize=9)
        
        fig.suptitle(f"{metric_display} - SIMD vs Non-SIMD Comparison (Excluding CUDA)", 
                    fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()
        
        output_path = metric_dir / f"{metric_name}_overview_simd_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
        plt.close()


def plot_variation_detail(df: pd.DataFrame, file_type: str, variation: str, 
                          metric_col: str, results_dir: Path) -> None:
    """
    Create detailed plot for a specific variation showing thread count scaling.
    
    Args:
        df: DataFrame with metrics data
        file_type: Type of files (accelerations or render)
        variation: Specific variation (e.g., pthread_blocked)
        metric_col: Name of the metric column
        results_dir: Directory to save plot image
    """
    # Filter for this specific variation
    df_var = df[(df["type"] == file_type) & 
                (df["variation"] == variation) & 
                (df[metric_col].notna())].copy()
    
    if len(df_var) == 0:
        return
    
    # Create subdirectory for this metric
    metric_name = metric_col.replace("_mean_ns", "")
    metric_dir = results_dir / metric_name
    metric_dir.mkdir(exist_ok=True)
    
    # Convert from ns to ms
    df_var[f"{metric_col}_ms"] = pd.to_numeric(df_var[metric_col], errors="coerce") / 1_000_000
    
    # Sort by thread count for proper ordering
    df_var["thread_num"] = df_var["thread_count"].apply(
        lambda x: int(x) if x and x.isdigit() else 0
    )
    df_var = df_var.sort_values("thread_num")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = range(len(df_var))
    ax.plot(x, df_var[f"{metric_col}_ms"], marker="o", linewidth=2, markersize=8, 
            color="tab:blue")
    
    # Configure plot
    metric_display = metric_name.replace("_", " ").title()
    variation_display = variation.replace("_", " ").title()
    
    # Create x-axis labels with thread counts
    labels = []
    for _, row in df_var.iterrows():
        if row["thread_count"]:
            labels.append(f"{row['thread_count']} threads")
        else:
            labels.append(row["csv_file_name"])
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Time (ms)", fontsize=12)
    ax.set_xlabel("Thread Count", fontsize=12)
    ax.set_title(f"{metric_display} - {variation_display}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    
    # Add value labels on points
    for i, (_, row) in enumerate(df_var.iterrows()):
        ax.annotate(f'{row[f"{metric_col}_ms"]:.1f}', 
                   xy=(i, row[f"{metric_col}_ms"]),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    output_filename = f"{file_type}_{variation}.png"
    output_path = metric_dir / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_all_plots(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Generate all plots: overviews and detailed variation plots.
    
    Args:
        df: DataFrame with metrics data
        results_dir: Directory to save plot images
    """
    metric_cols = [col for col in df.columns 
                   if col.endswith("_mean_ns") and col in METRICS_CONFIG]
    
    # 1. Generate overview plots for each metric
    print("\n=== Generating Metric Overview Plots ===")
    for metric_col in metric_cols:
        plot_metric_overview(df, metric_col, results_dir)
    
    # 2. Generate detailed plots for acceleration variations
    print("\n=== Generating Acceleration Variation Plots ===")
    df_acc = df[df["type"] == "accelerations"]
    acc_variations = df_acc["variation"].unique()
    
    for variation in sorted(acc_variations):
        # Plot both PhysicsStep and Accelerations for each variation
        for metric_col in ["physicsstep_mean_ns", "accelerations_mean_ns"]:
            if metric_col in df.columns:
                plot_variation_detail(df, "accelerations", variation, 
                                    metric_col, results_dir)
    
    # 3. Generate detailed plots for render variations
    print("\n=== Generating Render Variation Plots ===")
    df_render = df[df["type"] == "render"]
    render_variations = df_render["variation"].unique()
    
    for variation in sorted(render_variations):
        if "renderstep_mean_ns" in df.columns:
            plot_variation_detail(df, "render", variation, 
                                "renderstep_mean_ns", results_dir)


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics about the dataset."""
    print("\n=== Dataset Summary ===")
    print(f"Total files: {len(df)}")
    
    df_acc = df[df["type"] == "accelerations"]
    df_render = df[df["type"] == "render"]
    
    print(f"\nAcceleration files: {len(df_acc)}")
    print(f"  Unique variations: {df_acc['variation'].nunique()}")
    print(f"  Variations: {', '.join(sorted(df_acc['variation'].unique()))}")
    
    print(f"\nRender files: {len(df_render)}")
    print(f"  Unique variations: {df_render['variation'].nunique()}")
    print(f"  Variations: {', '.join(sorted(df_render['variation'].unique()))}")


def main() -> None:
    """Main execution function."""
    eval_dir = Path("parsed_eval")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate summary
    print("Extracting metrics from CSV files...")
    df = generate_summary(eval_dir)
    
    # Save summary
    summary_path = results_dir / "summary.csv"
    save_summary(df, summary_path)
    
    # Print statistics
    print_summary_stats(df)
    
    # Generate all visualizations
    generate_all_plots(df, results_dir)
    
    print(f"\n✓ All results saved to {results_dir}/")


if __name__ == "__main__":
    main()