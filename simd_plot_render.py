import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Load Data
# ============================================================
df = pd.read_csv('./results/plots/summary.csv')

# Only accelerations (physicsstep)
phys_df = df[df["type"] == "render"].copy()

# ============================================================
# SIMD Variants Only
# ============================================================
SIMD_VARIANTS = [
    "pthread_simd",
    "pthread",
    "serial",
    "serial_simd"
]

simd_df = phys_df[phys_df["variation"].isin(SIMD_VARIANTS)]

# Convert thread_count to int for sorting
simd_df["thread_count"] = simd_df["thread_count"].astype(int)

# Pretty labels
name_map = {
    "pthread_simd": "Pthread Simd",
    "pthread": "Pthread",
    "serial": "Serial",  
    "serial_simd": "Serial Simd"  
}

simd_df["label"] = simd_df["variation"].map(name_map)

# ============================================================
# Plot Mean Time
# ============================================================
plt.figure(figsize=(12, 7))

for var in SIMD_VARIANTS:
    data = simd_df[simd_df["variation"] == var].sort_values("thread_count")

    plt.plot(
        data["thread_count"],
        data["physicsstep_mean_ns"] / 1e6,  # Convert to milliseconds
        marker="o",
        linewidth=2,
        label=name_map[var]
    )

plt.xlabel("Thread Count")
plt.ylabel("Mean Time (ms)")
plt.title("SIMD Variations")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# ============================================================
# SAVE PNG
# ============================================================
plt.savefig("physicsstep_simd_mean_time.png", dpi=300)
print("Saved: physicsstep_simd_mean_time.png")

plt.show()
