"""Rebuild offline viewer data from the pinned anatomy and checked GPU simulation."""

import argparse
import base64
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from eat4_simulation import integrate, wiring

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("edges", type=Path)
args = parser.parse_args()
ids, chemical, electrical = wiring(args.edges)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
activity = integrate(ids, chemical, electrical, device, record_all=True)
report = json.loads((root / "eat4_result.json").read_text())
reference = torch.tensor([r["aib_activity"] for r in report["runs"]]).T
torch.testing.assert_close(
    activity[:, [ids["AIBL"], ids["AIBR"]]].mean(1), reference, rtol=1e-5, atol=1e-6
)
graph = nx.Graph()
graph.add_nodes_from(ids.values())
edges = []
with args.edges.open(newline="") as f:
    for row in csv.DictReader(f):
        a, b = (ids[row[k].strip()] for k in ("Source", "Target"))
        weight = float(row["Weight"])
        edges.append([a, b, weight, row["Type"].strip() == "electrical"])
        graph.add_edge(
            a, b, weight=weight + graph.get_edge_data(a, b, {}).get("weight", 0)
        )
layout = nx.spring_layout(graph, seed=42, iterations=100, method="force")
xy = np.array([layout[i] for i in range(len(ids))])
xy = (xy - xy.min(0)) / np.ptp(xy, axis=0)
# Quantize only display colors; the chart retains the original GPU readout.
colors = np.rint(
    np.log1p(activity.numpy().clip(0, 1) * 1000) / np.log1p(1000) * 255
).astype("uint8")
data = dict(
    names=list(ids),
    edges=edges,
    xy=xy.round(5).tolist(),
    report=report,
    colors=base64.b64encode(colors.tobytes()).decode(),
)
(root / "viewer_data.js").write_text(
    "const DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
)
