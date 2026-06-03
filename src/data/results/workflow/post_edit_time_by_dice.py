from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

WD = Path(__file__).resolve().parent

palette = {"HT7a": "green", "HT4": "orange", "HT30": "red"}
tablet_order = ["HT7a", "HT4", "HT30"]
method_markers = {"SAM": "o", "Classic": "s"}

dice_df = pd.read_csv(WD / "dice_scores.csv")
dice_df[["tablet", "method"]] = dice_df["tool_output"].str.rsplit(" ", n=1, expand=True)
dice_df = dice_df[dice_df["method"].isin(["SAM", "Classic"])].copy()

workflow_df = pd.read_csv(WD / "workflow.csv")
workflow_df = workflow_df[workflow_df["method"].isin(["SAM", "classic"])].copy()
workflow_df["method"] = workflow_df["method"].replace({"classic": "Classic"})
workflow_df["post_edit_time_minutes"] = pd.to_timedelta(workflow_df["time2"]).dt.total_seconds() / 60

plot_df = dice_df.merge(
    workflow_df[["tablet", "method", "post_edit_time_minutes"]],
    on=["tablet", "method"],
)

fig, ax = plt.subplots(figsize=(8, 5))

for tablet in tablet_order:
    for method, marker in method_markers.items():
        row = plot_df[(plot_df["tablet"] == tablet) & (plot_df["method"] == method)].iloc[0]
        ax.scatter(
            row["dice"],
            row["post_edit_time_minutes"],
            color=palette[tablet],
            edgecolor="black",
            marker=marker,
            s=110,
        )


# Add linear regression line trend (note: not a real regression due to small sample)
z = np.polyfit(plot_df["dice"], plot_df["post_edit_time_minutes"], 1)
p = np.poly1d(z)
x_line = np.linspace(plot_df["dice"].min(), plot_df["dice"].max(), 100)
ax.plot(x_line, p(x_line), linestyle="--", color="gray", linewidth=2)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", label="SAM", markerfacecolor="gray", markersize=10, markeredgecolor="black"),
    Line2D([0], [0], marker="s", color="w", label="Classic", markerfacecolor="gray", markersize=10, markeredgecolor="black"),
    Line2D([0], [0], marker="s",color="w", label="HT7a (easy)", markerfacecolor=palette["HT7a"], markersize=10, markeredgecolor="black"),
    Line2D([0], [0], marker="s", color="w", label="HT4 (medium)", markerfacecolor=palette["HT4"], markersize=10, markeredgecolor="black"),
    Line2D([0], [0], marker="s", color="w", label="HT30 (hard)", markerfacecolor=palette["HT30"], markersize=10, markeredgecolor="black"),
    Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label=f"Trend (n={len(plot_df)})"),
]
ax.legend(handles=legend_elements, fontsize=10, title_fontsize=12)



ax.set_xlabel("Dice",fontsize=12)
ax.set_ylabel("Post-editing time [min]",fontsize=12)
ax.set_title("Post-editing time by tool-export Dice score ",fontsize=16)
ax.set_ylim(0, plot_df["post_edit_time_minutes"].max() + 5)
ax.set_xlim(plot_df["dice"].min() - 0.05, plot_df["dice"].max() + 0.05)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
