import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df = pd.read_csv("overall_interactivity.csv")
tablet_order = ["HT7a", "HT4", "HT30"]

l = 225
d = 50
palette = {
    "box": (d/255, d/255, l/255),
    "bgd_points": (l/255, d/255, d/255),
    "fgd_points": (d/255, l/255, d/255)
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
        fontsize=16,
    )

    if boxes > 0:
        ax.text(
            position,
            boxes / 2 - 3,
            f"{boxes}",
            color="white",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    if bgd_points > 0:
        ax.text(
            position,
            boxes + bgd_points / 2 - 3,
            f"{bgd_points}",
            color="white",
            ha="center",
            va="bottom",
            fontsize=16,
        )

legend_elements = [
    Patch(facecolor=palette["fgd_points"], label="Foreground points"),
    Patch(facecolor=palette["bgd_points"], label="Background points"),
    Patch(facecolor=palette["box"], label="Boxes"),
]
legend = ax.legend(handles=legend_elements, loc="upper left",fontsize=18)
ax.add_artist(legend)
plt.yticks(fontsize=12)
plt.xticks(x, tablet_order, fontsize=16)
plt.xlabel("Tablet",fontsize=18)
plt.ylabel("Prompt count",fontsize=18)
plt.title("Prompt count per tablet",fontsize=18)
plt.legend(fontsize=18)
max_total = (df["boxes"] + df["bgd_points"] + df["fgd_points"]).max()
ax.set_ylim(0, max_total * 1.5)
plt.tight_layout()
plt.show()