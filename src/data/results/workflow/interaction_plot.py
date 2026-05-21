import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df = pd.read_csv("overall_interactivity.csv")
tablet_order = ["HT7a", "HT4", "HT30"]

palette = {
    "box": "blue",
    "bgd_points": "red",
    "fgd_points": "green",
}

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(tablet_order))
width = 0.5

for i, tablet in enumerate(tablet_order):
    tablet_df = df[df["tablet"] == tablet]
    boxes = tablet_df["boxes"].iloc[0]
    bgd_points = tablet_df["bgd_points"].iloc[0]
    fgd_points = tablet_df["fgd_points"].iloc[0]
    position = x[i]

    plt.bar(
        position,
        boxes,
        width=width,
        color=palette["box"],
        alpha=1,
    )

    plt.bar(
        position,
        bgd_points,
        width=width,
        bottom=boxes,
        color=palette["bgd_points"],
        alpha=1,
    )

    plt.bar(
        position,
        fgd_points,
        width=width,
        bottom=boxes + bgd_points,
        color=palette["fgd_points"],
        alpha=1,
    )

    total = boxes + bgd_points + fgd_points

    ax.text(
        position,
        total + 1,
        f"{total}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    if boxes > 0:
        ax.text(
            position,
            boxes / 2 - 1,
            f"{boxes}",
            color="white",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    if bgd_points > 0:
        ax.text(
            position,
            boxes + bgd_points / 2 - 1,
            f"{bgd_points}",
            color="white",
            ha="center",
            va="bottom",
            fontsize=10,
        )

legend_elements = [
    Patch(facecolor=palette["fgd_points"], label="Foreground points"),
    Patch(facecolor=palette["bgd_points"], label="Background points"),
    Patch(facecolor=palette["box"], label="Boxes"),
]
legend = ax.legend(handles=legend_elements, loc="upper left")
ax.add_artist(legend)
plt.xticks(x, tablet_order)
plt.xlabel("Tablet")
plt.ylabel("Prompt count")
plt.title("Ester's prompt count per tablet")
plt.legend()
max_total = (df["boxes"] + df["bgd_points"] + df["fgd_points"]).max()
ax.set_ylim(0, max_total * 1.3)
plt.tight_layout()
plt.show()