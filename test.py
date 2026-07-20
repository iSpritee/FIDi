from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


timesteps = np.array([800, 600, 400, 300, 250, 200, 100, 0])
positions = np.arange(len(timesteps))

# Each timestep is calculated from 1,000 samples. Generate this array from
# timesteps so its length stays consistent when more timesteps are added.
counts = np.full(timesteps.shape, 1000, dtype=int)

lfer_mean = np.array([
    0.925222, 
    0.918813,
    0.903819,
    0.893043,
    0.887583,
    0.882105,
    0.871926,
    0.859783
])

lfer_std = np.array([
    0.054459,
    0.057626,
    0.063125,
    0.069832,
    0.072780,
    0.075362,
    0.080015,
    0.082814 
])

centroid_mean = np.array([
    0.044129,
    0.047349,
    0.051383,
    0.053596,
    0.054563,
    0.055497,
    0.057325,
    0.059517
])

centroid_std = np.array([
    0.011132,
    0.012377,
    0.013501,
    0.014587,
    0.015016,
    0.015386,
    0.016080,
    0.016384
])

data_arrays = {
    "counts": counts,
    "lfer_mean": lfer_mean,
    "lfer_std": lfer_std,
    "centroid_mean": centroid_mean,
    "centroid_std": centroid_std,
}
for name, values in data_arrays.items():
    if len(values) != len(timesteps):
        raise ValueError(
            f"{name} has {len(values)} values, but timesteps has "
            f"{len(timesteps)} values."
        )


lfer_ci95 = 1.96 * lfer_std / np.sqrt(counts)
centroid_ci95 = 1.96 * centroid_std / np.sqrt(counts)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

save_dir = Path("./spectral_figures")
save_dir.mkdir(parents=True, exist_ok=True)


def add_value_labels(ax, x, y, offset, va):
    """Add numerical values near the data points."""
    for x_i, y_i in zip(x, y):
        ax.text(
            x_i,
            y_i + offset,
            f"{y_i:.3f}",
            ha="center",
            va=va,
            fontsize=9,
        )


# ============================================================
# LFER and spectral centroid subplots
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax_lfer, ax_centroid = axes

ax_lfer.errorbar(
    positions,
    lfer_mean,
    yerr=lfer_ci95,
    color="tab:blue",
    marker="o",
    markersize=6,
    linewidth=2,
    capsize=4,
    capthick=1.2,
)

ax_centroid.errorbar(
    positions,
    centroid_mean,
    yerr=centroid_ci95,
    color="tab:orange",
    marker="s",
    markersize=6,
    linewidth=2,
    capsize=4,
    capthick=1.2,
)

for ax in axes:
    ax.set_xticks(positions)
    ax.set_xticklabels([str(t) for t in timesteps])
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.35,
    )
    ax.set_xlim(-0.25, len(positions) - 0.75)

lfer_range = np.ptp(lfer_mean)
ax_lfer.set_ylim(
    lfer_mean.min() - 0.15 * lfer_range,
    lfer_mean.max() + 0.25 * lfer_range,
)
ax_lfer.set_title("Low-frequency energy ratio")
add_value_labels(
    ax_lfer,
    positions,
    lfer_mean,
    offset=0.06 * lfer_range,
    va="bottom",
)

centroid_range = np.ptp(centroid_mean)
ax_centroid.set_ylim(
    centroid_mean.min() - 0.15 * centroid_range,
    centroid_mean.max() + 0.25 * centroid_range,
)
ax_centroid.set_title("Spectral centroid")
add_value_labels(
    ax_centroid,
    positions,
    centroid_mean,
    offset=0.06 * centroid_range,
    va="bottom",
)

fig.tight_layout()

fig.savefig(
    save_dir / "lfer_centroid_denoising_curve.pdf",
    bbox_inches="tight",
)

fig.savefig(
    save_dir / "lfer_centroid_denoising_curve.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Figures saved to: {save_dir.resolve()}")
