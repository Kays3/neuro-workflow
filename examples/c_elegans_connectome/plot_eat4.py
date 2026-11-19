"""Render the paired AIB traces from the saved GPU simulation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).parent
result = json.loads((root / "eat4_result.json").read_text())
fig, ax = plt.subplots(figsize=(7.8, 4.4), layout="constrained")
colors = {1.0: "#126e82", 0.5: "#94601b", 0.0: "#9b3f64"}
for run in result["runs"]:
    if run["gain"] == 0.8:
        y = run["aib_activity"]
        release = run["ash_release"]
        ax.plot(
            [i * result["dt"] for i in range(len(y))],
            y,
            lw=2.5,
            color=colors[release],
            label=f"{release:.0%} ASH chemical release",
        )
ax.axvspan(*result["ash_pulse"], color="#607386", alpha=0.12, label="ASH stimulus")
ax.set(
    title="Simulated AIB response to an ASH pulse",
    xlabel="Time (model units)",
    ylabel="Mean AIBL and AIBR activity (model units)",
    xlim=(0, 18),
    ylim=(0, None),
)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "eat4_response.png", dpi=170)
