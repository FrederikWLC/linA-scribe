import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log_df = pd.read_csv("interaction_log.csv")
workflow_df = pd.read_csv("workflow.csv")

run_df = (
    log_df[(log_df["action"] == "decode_mask") | (log_df["action"] == "set_image")]
    .assign(duration=lambda d: pd.to_timedelta(d["timestamp_end"]) - pd.to_timedelta(d["timestamp_start"]))
)


run_summary = (
    run_df
    .groupby("method", as_index=False)["duration"]
    .sum()
    .rename(columns={"duration": "action_duration"})
)

workflow_df["time1"] = pd.to_timedelta(workflow_df["time1"])
workflow_summary = (
    workflow_df[workflow_df["method"].isin(["classic", "SAM"])]
    .groupby("method", as_index=False)["time1"]
    .sum()
    .rename(columns={"time1": "tool_duration"})
)
workflow_summary["tool_minutes"] = workflow_summary["tool_duration"].dt.total_seconds() / 60

compare_df = (
    workflow_summary
    .merge(run_summary, on="method", how="outer")
    .fillna({"tool_duration": pd.Timedelta(0), "action_duration": pd.Timedelta(0)})
    .assign(
        run_minutes=lambda d: d["action_duration"].dt.total_seconds() / 60,
    )
)
compare_df["tool_rest_minutes"] = compare_df["tool_minutes"] - compare_df["run_minutes"]

method_order = ["classic", "SAM"]
compare_df = compare_df.set_index("method").loc[method_order].reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]})

x = np.arange(len(method_order))
width = 0.5

ax1.bar(
    x,
    compare_df["run_minutes"],
    width=width,
    label="Model execution time",
    color="darkred",
    alpha=0.5
)
ax1.bar(
    x,
    compare_df["tool_rest_minutes"],
    width=width,
    bottom=compare_df["run_minutes"],
    color="darkgrey",
    alpha=0.25
)

ax1.set_ylim(0, compare_df["tool_minutes"].max() * 1.2)

for xi, row in compare_df.iterrows():
    total_minutes = row["run_minutes"] + row["tool_rest_minutes"]
    ax1.text(
        x[xi],
        total_minutes + 0.15,
        f"{row['tool_duration'].components.minutes}:{row['tool_duration'].components.seconds:02d}",
        ha="center",
        va="bottom",
        fontsize=16,
    )
    if row["run_minutes"] >= 1:
        ax1.text(
            x[xi],
            row["run_minutes"] + 0.15,
            f"{row['action_duration'].components.minutes}:{row['action_duration'].components.seconds:02d}",
            ha="center",
            va="bottom",
            fontsize=16,
            color="black",
        )

ax1.set_xticks(x)
ax1.set_xticklabels(method_order, fontsize=18)
ax1.set_xlabel("Method", fontsize=20)
ax1.set_ylabel("Time [min]", fontsize=18)
ax1.set_title(
    "Tool time, including model execution",
    fontsize=20,
)
ax1.tick_params(axis="both", which="major", labelsize=18)
ax1.legend(fontsize=18, loc="upper left")

sam_run_df = run_df[run_df["method"] == "SAM"]
sam_run_seconds = sam_run_df["duration"].dt.total_seconds()
bins = np.linspace(0, sam_run_seconds.max() * 1.05, 12)
counts, bin_edges = np.histogram(sam_run_seconds, bins=bins, density=False)
relative_counts = counts / counts.sum()
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

ax2.barh(bin_centers, relative_counts, height=np.diff(bin_edges), alpha=0.5, color="darkred", edgecolor="black")
ax2.set_ylabel("Time [s]", fontsize=18)
ax2.set_xlabel("Relative frequency [%]", fontsize=20)
ax2.set_title("Distribution of SAM Modal execution time", fontsize=20)
ax2.set_yticks(np.arange(0, int(sam_run_seconds.max() * 1.05) + 1, 5))
ax2.tick_params(axis="both", which="major", labelsize=16)

plt.tight_layout()
plt.show()
