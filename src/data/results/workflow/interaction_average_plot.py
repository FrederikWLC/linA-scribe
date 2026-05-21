import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("interactivity_per_box.csv")
df["tablet"] = pd.Categorical(df["tablet"], categories=["HT7a", "HT4", "HT30"], ordered=True)

tablet_order = ["HT7a", "HT4", "HT30"]

fig, ax = plt.subplots(figsize=(8, 5))

palette = {"HT7a": "green", "HT4": "orange", "HT30": "red"}

sns.swarmplot(
    data=df,
    x="tablet",
    y="bgd_points",
    
    order=tablet_order,
    size=7,
    ax=ax,
    hue="tablet",
    palette=palette,
)

for i, tablet in enumerate(tablet_order):
    tablet_df = df[df["tablet"] == tablet]
    mean_points = tablet_df["bgd_points"].mean()
    ax.hlines(
        y=mean_points,
        xmin=i - 0.18,
        xmax=i + 0.18,
        color=palette[tablet],
        linewidth=4,
        alpha=0.4,
        zorder=10
    )

    text_coords = {
        "HT7a": (i+0.4, mean_points-0.15),
        "HT4": (i-0.4, mean_points-0.15),
        "HT30": (i-0.4, mean_points-0.15),
    }

    ax.text(
        text_coords[tablet][0],
        text_coords[tablet][1],
        f"mean = {mean_points:.1f}",
        color=palette[tablet],
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    

ax.set_xlabel("Tablet")
ax.set_ylabel("Point prompts / box")
ax.set_title("Distribution of SAM point prompts per box")
plt.tight_layout()
plt.show()
