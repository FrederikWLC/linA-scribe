from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
workflow_df["tool_time_minutes"] = pd.to_timedelta(workflow_df["time1"]).dt.total_seconds() / 60

plot_df = dice_df.merge(
    workflow_df[["tablet", "method", "tool_time_minutes"]],
    on=["tablet", "method"],
)

fig, ax = plt.subplots(figsize=(8, 5))

for tablet in tablet_order:
    for method, marker in method_markers.items():
        row = plot_df[(plot_df["tablet"] == tablet) & (plot_df["method"] == method)].iloc[0]
        ax.scatter(
            row["tool_time_minutes"],
            row["dice"],
            color=palette[tablet],
            edgecolor="black",
            marker=marker,
            s=110,
        )
        ax.text(
            row["tool_time_minutes"]-0.12,
            row["dice"] + 0.0075,
            f"{tablet} {method}",
            va="center",
            fontsize=8,
        )

ax.set_xlabel("Tool time [min]")
ax.set_ylabel("Dice")
ax.set_title("Tool-export Dice score by tool time")
ax.set_ylim(plot_df["dice"].min() - 0.01, plot_df["dice"].max() + 0.02)
ax.set_xlim(0, plot_df["tool_time_minutes"].max() + 1)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
