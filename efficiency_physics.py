
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./results/plots/summary.csv')
phys_df = df[df["type"] == "accelerations"].copy()

# ============================================================
# Find baselines
# ============================================================
baseline_normal = phys_df[(phys_df["variation"] == "serial") & (phys_df["thread_count"] == 1)]
baseline_simd = phys_df[(phys_df["variation"] == "serial_simd") & (phys_df["thread_count"] == 1)]

if baseline_normal.empty or baseline_simd.empty:
    raise ValueError("Missing serial or serial_simd baseline")

normal_base_time = baseline_normal.iloc[0]["physicsstep_mean_ns"]
simd_base_time = baseline_simd.iloc[0]["physicsstep_mean_ns"]

# ============================================================
# Assign which baseline each row should use
# ============================================================
SIMD_VARIANTS = [
    "pthread_simd_blocked",
    "pthread_simd_interleaved",
    "pthread_mutex_simd_blocked",
    "pthread_mutex_simd_interleaved",
    "serial_simd"
]

def pick_baseline(row):
    if row["variation"] in SIMD_VARIANTS:
        return simd_base_time
    else:
        return normal_base_time

phys_df["baseline_ns"] = phys_df.apply(pick_baseline, axis=1)

# ============================================================
# Compute speedup & efficiency
# ============================================================
phys_df["physicsstep_speedup"] = phys_df["baseline_ns"] / phys_df["physicsstep_mean_ns"]
phys_df["physicsstep_efficiency"] = phys_df["physicsstep_speedup"] / phys_df["thread_count"]

# Label cleanup
phys_df["label"] = phys_df["variation"].str.replace("accelerations_", "")

# Exclude baseline rows from plotting
phys_df = phys_df[phys_df["thread_count"] > 1]

# ============================================================
# Prepare Data
# ============================================================
variations = phys_df["label"].unique()
threads = sorted(phys_df["thread_count"].unique())

x = np.arange(len(threads))
bar_width = 0.8 / len(variations)

# ============================================================
# Plot 1: Speedup
# ============================================================
plt.figure(figsize=(14, 8))

for i, var in enumerate(variations):
    data = phys_df[phys_df["label"] == var].sort_values("thread_count")
    plt.bar(x + i * bar_width,
            data["physicsstep_speedup"],
            width=bar_width,
            label=var.replace("_", " "))

plt.xticks(x + bar_width * len(variations) / 2, threads)
plt.xlabel("Thread Count")
plt.ylabel("Speedup (×)")
plt.title("Physicsstep Speedup by Threads")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("physicsstep_speedup.png")
# plt.show

# csv output with columns are thread_count, and rows are variations, values to 4 decimal places
speedup_table = phys_df.pivot(index="label", columns="thread_count", values="physicsstep_speedup")
# speedup_table = speedup_table.sort_index()
speedup_table = speedup_table.round(4)
speedup_table.to_csv("physicsstep_speedup_table.csv") 

# ============================================================
# Plot 2: Efficiency
# ============================================================
plt.figure(figsize=(14, 8))

for i, var in enumerate(variations):
    data = phys_df[phys_df["label"] == var].sort_values("thread_count")
    plt.bar(x + i * bar_width,
            data["physicsstep_efficiency"],
            width=bar_width,
            label=var.replace("_", " "))

plt.xticks(x + bar_width * len(variations) / 2, threads)
plt.xlabel("Thread Count")
plt.ylabel("Efficiency (speedup / threads)")
plt.title("Physicsstep Efficiency by Threads")
plt.ylim(0, 1.2)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("physicsstep_efficiency.png")

# csv output with columns are thread_count, and rows are variations, values to 4 decimal places
efficiency_table = phys_df.pivot(index="label", columns="thread_count", values="physicsstep_efficiency")
# efficiency_table = efficiency_table.sort_index()
efficiency_table = efficiency_table.round(4)
efficiency_table.to_csv("physicsstep_efficiency_table.csv")