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


def metric_base_name(metric_col: str) -> str:
    """Strip common suffixes to get the logical metric base name."""
    for suffix in ("_mean_ns", "_speedup", "_efficiency"):
        if metric_col.endswith(suffix):
            return metric_col[: -len(suffix)]
    return metric_col


def is_simd_variation(variation: str | None) -> bool:
    """Return True if the variation name denotes SIMD usage."""
    return bool(isinstance(variation, str) and "simd" in variation.lower())


def thread_count_to_int(thread_count: str | None) -> int:
    """Convert a thread count string to int, defaulting to 1 for serial runs."""
    if isinstance(thread_count, str) and thread_count.isdigit():
        return int(thread_count)
    return 1


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


def add_speedup_and_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add speedup and efficiency columns for each runtime metric.
    
    Speedup is computed against a serial baseline of the same type.
    SIMD variations prefer the serial_simd baseline when available.
    """
    df = df.copy()
    runtime_cols = [col for col in df.columns 
                    if col.endswith("_mean_ns") and col in METRICS_CONFIG]
    
    for runtime_col in runtime_cols:
        speedup_col = runtime_col.replace("_mean_ns", "_speedup")
        efficiency_col = runtime_col.replace("_mean_ns", "_efficiency")
        df[speedup_col] = pd.NA
        df[efficiency_col] = pd.NA
        
        for file_type in df["type"].dropna().unique():
            df_type = df[df["type"] == file_type]
            
            # Determine baselines for SIMD and non-SIMD variations
            baselines = {}
            for simd_flag in (False, True):
                base_variation = "serial_simd" if simd_flag else "serial"
                baseline_rows = df_type[
                    (df_type["variation"] == base_variation) & 
                    (df_type[runtime_col].notna())
                ]
                if not baseline_rows.empty:
                    baseline_val = pd.to_numeric(
                        baseline_rows.iloc[0][runtime_col], errors="coerce"
                    )
                    if pd.notna(baseline_val):
                        baselines[simd_flag] = baseline_val
            
            # Prefer matching SIMD baseline, otherwise fall back to any available baseline
            default_baseline = baselines.get(False) or baselines.get(True)
            
            for idx, row in df_type[df_type[runtime_col].notna()].iterrows():
                runtime_ns = pd.to_numeric(row[runtime_col], errors="coerce")
                if pd.isna(runtime_ns) or runtime_ns <= 0:
                    continue
                
                simd_flag = is_simd_variation(row["variation"])
                baseline = baselines.get(simd_flag, default_baseline)
                if baseline is None or baseline <= 0:
                    continue
                
                speedup = baseline / runtime_ns
                threads = thread_count_to_int(row["thread_count"])
                efficiency = speedup / threads if threads > 0 else pd.NA
                
                df.at[idx, speedup_col] = speedup
                df.at[idx, efficiency_col] = efficiency
    
    return df


def save_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Save summary DataFrame to CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Summary written to {output_path}")
    print(f"Processed {len(df)} CSV files")


def plot_metric_overview(
    df: pd.DataFrame,
    metric_col: str,
    results_dir: Path,
    y_label: str,
    title_suffix: str,
    convert_to_ms: bool = False,
) -> None:
    """
    Create overview plots for a single metric:
    1. All variations grouped by SIMD/non-SIMD (excluding CUDA)
    2. All variations as separate lines
    
    Args:
        df: DataFrame with metrics data
        metric_col: Name of the metric column
        results_dir: Directory to save plot image
        y_label: Label for the Y-axis
        title_suffix: Text to append after the metric name in plot titles
        convert_to_ms: If True, convert values from ns to ms
    """
    df_metric = df[df[metric_col].notna()].copy()
    
    if len(df_metric) == 0:
        return
    
    metric_name = metric_base_name(metric_col)
    metric_dir = results_dir / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    df_metric["plot_value"] = pd.to_numeric(df_metric[metric_col], errors="coerce")
    if convert_to_ms:
        df_metric["plot_value"] = df_metric["plot_value"] / 1_000_000
    
    df_metric["thread_num"] = df_metric["thread_count"].apply(thread_count_to_int)
    df_metric["is_simd"] = df_metric["variation"].apply(is_simd_variation)
    df_metric["is_cuda"] = df_metric["variation"].str.contains("cuda", case=False, na=False)
    
    variations = df_metric["variation"].unique()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for variation in sorted(variations):
        df_var = df_metric[df_metric["variation"] == variation].copy()
        df_var = df_var.sort_values("thread_num")
        
        x_values = [thread_count_to_int(row["thread_count"]) for _, row in df_var.iterrows()]
        label = variation.replace("_", " ").title()
        ax.plot(
            x_values,
            df_var["plot_value"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=label,
        )
    
    metric_display = metric_name.replace("_", " ").title()
    suffix_text = f" {title_suffix}" if title_suffix else ""
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("Thread Count", fontsize=12)
    ax.set_title(f"{metric_display}{suffix_text} - All Variations", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    
    plt.tight_layout()
    output_path = metric_dir / f"{metric_name}_overview_all.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()
    
    df_no_cuda = df_metric[~df_metric["is_cuda"]].copy()
    
    if len(df_no_cuda) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        df_non_simd = df_no_cuda[~df_no_cuda["is_simd"]]
        for variation in sorted(df_non_simd["variation"].unique()):
            df_var = df_non_simd[df_non_simd["variation"] == variation].copy()
            df_var = df_var.sort_values("thread_num")
            
            x_values = [thread_count_to_int(row["thread_count"]) for _, row in df_var.iterrows()]
            label = variation.replace("_", " ").title()
            ax1.plot(
                x_values,
                df_var["plot_value"],
                marker="o",
                linewidth=2,
                markersize=6,
                label=label,
            )
        
        ax1.set_ylabel(y_label, fontsize=12)
        ax1.set_xlabel("Thread Count", fontsize=12)
        ax1.set_title("Non-SIMD Variations", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(fontsize=9)
        
        df_simd = df_no_cuda[df_no_cuda["is_simd"]]
        for variation in sorted(df_simd["variation"].unique()):
            df_var = df_simd[df_simd["variation"] == variation].copy()
            df_var = df_var.sort_values("thread_num")
            
            x_values = [thread_count_to_int(row["thread_count"]) for _, row in df_var.iterrows()]
            label = variation.replace("_", " ").title()
            ax2.plot(
                x_values,
                df_var["plot_value"],
                marker="o",
                linewidth=2,
                markersize=6,
                label=label,
            )
        
        ax2.set_ylabel(y_label, fontsize=12)
        ax2.set_xlabel("Thread Count", fontsize=12)
        ax2.set_title("SIMD Variations", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(fontsize=9)
        
        fig.suptitle(
            f"{metric_display}{suffix_text} - SIMD vs Non-SIMD Comparison (Excluding CUDA)",
            fontsize=14,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()
        
        output_path = metric_dir / f"{metric_name}_overview_simd_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
        plt.close()


def plot_variation_detail(
    df: pd.DataFrame,
    file_type: str,
    variation: str,
    metric_col: str,
    results_dir: Path,
    y_label: str,
    title_suffix: str,
    convert_to_ms: bool = False,
    value_format: str = ".1f",
) -> None:
    """
    Create detailed plot for a specific variation showing thread count scaling.
    
    Args:
        df: DataFrame with metrics data
        file_type: Type of files (accelerations or render)
        variation: Specific variation (e.g., pthread_blocked)
        metric_col: Name of the metric column
        results_dir: Directory to save plot image
        y_label: Label for the Y-axis
        title_suffix: Text to append after the metric name in plot titles
        convert_to_ms: If True, convert values from ns to ms
        value_format: Format string for value annotations
    """
    df_var = df[
        (df["type"] == file_type)
        & (df["variation"] == variation)
        & (df[metric_col].notna())
    ].copy()
    
    if len(df_var) == 0:
        return
    
    metric_name = metric_base_name(metric_col)
    metric_dir = results_dir / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    df_var["plot_value"] = pd.to_numeric(df_var[metric_col], errors="coerce")
    if convert_to_ms:
        df_var["plot_value"] = df_var["plot_value"] / 1_000_000
    
    df_var["thread_num"] = df_var["thread_count"].apply(thread_count_to_int)
    df_var = df_var.sort_values("thread_num")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = range(len(df_var))
    ax.plot(
        x,
        df_var["plot_value"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="tab:blue",
    )
    
    metric_display = metric_name.replace("_", " ").title()
    variation_display = variation.replace("_", " ").title()
    suffix_text = f" {title_suffix}" if title_suffix else ""
    
    labels = []
    for _, row in df_var.iterrows():
        if row["thread_count"]:
            labels.append(f"{row['thread_count']} threads")
        else:
            labels.append(row["csv_file_name"])
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("Thread Count", fontsize=12)
    ax.set_title(f"{metric_display}{suffix_text} - {variation_display}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    
    for i, (_, row) in enumerate(df_var.iterrows()):
        value_text = f"{row['plot_value']:{value_format}}"
        ax.annotate(
            value_text,
            xy=(i, row["plot_value"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            alpha=0.7,
        )
    
    plt.tight_layout()
    
    output_filename = f"{file_type}_{variation}.png"
    output_path = metric_dir / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def metric_column_for_mode(metric_base: str, mode: str) -> str:
    """Return the column name for a metric base under the selected plotting mode."""
    suffix_map = {
        "runtime": "_mean_ns",
        "speedup": "_speedup",
        "efficiency": "_efficiency",
    }
    return f"{metric_base}{suffix_map[mode]}"


def generate_all_plots(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Generate runtime, speedup, and efficiency plots.
    
    Args:
        df: DataFrame with metrics data
        results_dir: Directory to save plot images
    """
    plot_modes = {
        "runtime": {
            "suffix": "Runtime",
            "y_label": "Mean Time (ms)",
            "convert_to_ms": True,
            "value_format": ".1f",
        },
        "speedup": {
            "suffix": "Speedup",
            "y_label": "Speedup (×)",
            "convert_to_ms": False,
            "value_format": ".2f",
        },
        "efficiency": {
            "suffix": "Efficiency",
            "y_label": "Efficiency",
            "convert_to_ms": False,
            "value_format": ".2f",
        },
    }
    
    metric_bases = sorted({metric_base_name(col) for col in METRICS_CONFIG})
    type_metric_bases = {
        "accelerations": ["physicsstep", "accelerations"],
        "render": ["renderstep"],
    }
    
    for mode, config in plot_modes.items():
        mode_dir = results_dir / mode
        mode_dir.mkdir(exist_ok=True)
        
        metric_cols = [
            metric_column_for_mode(base, mode)
            for base in metric_bases
            if metric_column_for_mode(base, mode) in df.columns
        ]
        
        print(f"\n=== Generating {config['suffix']} Overview Plots ===")
        for metric_col in metric_cols:
            plot_metric_overview(
                df,
                metric_col,
                mode_dir,
                y_label=config["y_label"],
                title_suffix=config["suffix"],
                convert_to_ms=config["convert_to_ms"],
            )
        
        print(f"\n=== Generating {config['suffix']} Acceleration Variation Plots ===")
        df_acc = df[df["type"] == "accelerations"]
        acc_variations = df_acc["variation"].dropna().unique()
        for variation in sorted(acc_variations):
            for metric_base in type_metric_bases["accelerations"]:
                metric_col = metric_column_for_mode(metric_base, mode)
                if metric_col in df.columns:
                    plot_variation_detail(
                        df,
                        "accelerations",
                        variation,
                        metric_col,
                        mode_dir,
                        y_label=config["y_label"],
                        title_suffix=config["suffix"],
                        convert_to_ms=config["convert_to_ms"],
                        value_format=config["value_format"],
                    )
        
        print(f"\n=== Generating {config['suffix']} Render Variation Plots ===")
        df_render = df[df["type"] == "render"]
        render_variations = df_render["variation"].dropna().unique()
        for variation in sorted(render_variations):
            metric_col = metric_column_for_mode("renderstep", mode)
            if metric_col in df.columns:
                plot_variation_detail(
                    df,
                    "render",
                    variation,
                    metric_col,
                    mode_dir,
                    y_label=config["y_label"],
                    title_suffix=config["suffix"],
                    convert_to_ms=config["convert_to_ms"],
                    value_format=config["value_format"],
                )


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
    df = add_speedup_and_efficiency(df)
    
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
