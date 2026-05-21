import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

df = pd.read_csv("workflow.csv")

df["time1"] = pd.to_timedelta(df["time1"])
df["time2"] = pd.to_timedelta(df["time2"])
df["time"] = df["time1"] + df["time2"]
df["time1_minutes"] = df["time1"].dt.total_seconds() / 60
df["time2_minutes"] = df["time2"].dt.total_seconds() / 60
df["time_minutes"] = df["time"].dt.total_seconds() / 60

fig, ax = plt.subplots(figsize=(10, 6))

method_order = ["manual", "classic", "SAM"]
tablet_order = ["HT7a", "HT4", "HT30"]

palette = {
    "HT7a": "green",
    "HT4": "orange",
    "HT30": "red",
}

x = np.arange(len(method_order))
width = 0.25

for i, tablet in enumerate(tablet_order):
    method_df = (
        df[df["tablet"] == tablet]
        .set_index("method")
        .loc[method_order]
    )

    positions = x + (i - 1) * width

    plt.bar(
        positions,
        method_df["time1_minutes"],
        width=width,
        color=palette[tablet],
        alpha=1,
    )

    plt.bar(
        positions,
        method_df["time2_minutes"],
        width=width,
        bottom=method_df["time1_minutes"],
        color=palette[tablet],
        alpha=0.3,
    )

    for pos, method in zip(positions, method_order):
        tool = method_df.loc[method, "time1"]
        post = method_df.loc[method, "time2"]
        total = method_df.loc[method, "time"]
        tool_minutes = method_df.loc[method, "time1_minutes"]
        post_minutes = method_df.loc[method, "time2_minutes"]
        total_minutes = tool_minutes + post_minutes

        ax.text(
            pos,
            total_minutes + 0.25,
            f"{total.components.minutes}:{total.components.seconds:02d}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        if tool_minutes > 0:
            ax.text(
                pos,
                tool_minutes + 0.15,
                f"{tool.components.minutes}:{tool.components.seconds:02d}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

plt.xticks(x, method_order)
plt.xlabel("Method")
plt.ylabel("Time [min]")
plt.title("Ester's workflow time spent per method, split into tool use and post-editing")
legend1 = [
    Patch(facecolor="black", alpha=0.3, label="Krita post-editing time"),
    Patch(facecolor="black", alpha=1, label="Tool use time"),
]
legend2 = [
Patch(facecolor="green", alpha=1, label="HT7a (easy)"),
    Patch(facecolor="orange", alpha=1, label="HT4 (medium)"),
    Patch(facecolor="red", alpha=1, label="HT30 (hard)"),
]
legend1_obj = ax.legend(handles=legend1, loc="lower left")
ax.add_artist(legend1_obj)
ax.legend(handles=legend2, loc="upper left")

plt.tight_layout()
plt.show()
