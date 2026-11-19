"""Simulate ASH release and bilateral AIB activity on a GPU via NeuroWorkflow."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import torch
from neuroworkflow import (
    MethodDefinition,
    Node,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
    PortType,
    WorkflowBuilder,
)

EDGE_HASH = "142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca"
RELEASE = (1.0, 0.5, 0.0)
GAINS = (0.4, 0.8, 1.2)
DT = 0.1
PULSE = (2.0, 6.0)


def wiring(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != EDGE_HASH:
        raise ValueError("Unexpected OpenWorm edge-list hash")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = sorted({row[k].strip() for row in rows for k in ("Source", "Target")})
    ids = {name: i for i, name in enumerate(names)}
    matrices = {
        kind: torch.zeros(len(ids), len(ids)) for kind in ("chemical", "electrical")
    }
    for row in rows:
        target, source = ids[row["Target"].strip()], ids[row["Source"].strip()]
        matrices[row["Type"].strip()][target, source] += float(row["Weight"])
    chem, gap = matrices.values()
    chem.fill_diagonal_(0)
    gap = (gap + gap.T) / 2
    gap.fill_diagonal_(0)
    scale = (chem + gap).sum(dim=1, keepdim=True).clamp_min(1)
    return ids, chem / scale, gap / scale


@torch.inference_mode()
def integrate(ids, chemical, electrical, device):
    # One column per condition, with paired parameters and a common stimulus.
    conditions = [(gain, release) for gain in GAINS for release in RELEASE]
    gain, release = torch.tensor(conditions, device=device).T
    ash = [ids["ASHL"], ids["ASHR"]]
    aib = [ids["AIBL"], ids["AIBR"]]
    c, e = chemical.to(device), electrical.to(device)
    state = torch.zeros(len(ids), len(conditions), device=device)
    source_scale = torch.ones_like(state)
    source_scale[ash] = release.expand(2, -1)
    drive = torch.zeros_like(state)
    drive[ash] = 1
    gap_degree = e.sum(dim=1, keepdim=True)
    trace = [state[aib].mean(dim=0)]
    for step in range(180):
        chemical_input = c @ (source_scale * state)
        gap_input = e @ state - gap_degree * state
        stimulus = drive if PULSE[0] <= step * DT < PULSE[1] else 0
        state = state + DT * (
            -state + torch.tanh(stimulus + gain * (chemical_input + gap_input))
        )
        trace.append(state[aib].mean(dim=0))
    return torch.stack(trace).cpu()


class Eat4CircuitNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="eat4_ash_circuit",
        stage="simulation",
        tool="PyTorch",
        model_source="https://doi.org/10.1038/s41467-023-44638-5",
        description="Simulate how assumed ASH chemical release affects bilateral AIB activity.",
        parameters={
            "edges": ParameterDefinition(
                default_value="edges.csv", description="Pinned OpenWorm edge list."
            )
        },
        inputs={},
        outputs={
            "report": PortDefinition(
                type=PortType.DICT, description="GPU traces and CPU agreement."
            )
        },
        methods={
            "simulate": MethodDefinition(
                description="Integrate matched release conditions.",
                inputs=[],
                outputs=["report"],
            )
        },
    )

    def __init__(self, name):
        super().__init__(name)
        self.add_process_step("simulate", self.simulate, method_key="simulate")

    def simulate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        ids, chem, gap = wiring(Path(self._parameters["edges"]))
        traces = integrate(ids, chem, gap, device)
        cpu = integrate(ids, chem, gap, torch.device("cpu"))
        torch.testing.assert_close(traces, cpu, atol=1e-6, rtol=1e-5)
        if (
            not traces.isfinite().all()
            or traces[:21].count_nonzero()
            or traces.max() <= 0
        ):
            raise RuntimeError("Unexpected baseline or evoked activity")
        runs = []
        for i, (gain, release) in enumerate((g, r) for g in GAINS for r in RELEASE):
            runs.append(
                dict(
                    gain=gain,
                    ash_release=release,
                    peak_aib=traces[:, i].max().item(),
                    aib_activity=traces[:, i].tolist(),
                )
            )
        return {
            "report": dict(
                device=str(device),
                machine=platform.machine(),
                torch=str(torch.__version__),
                edge_sha256=EDGE_HASH,
                nodes=len(ids),
                dt=DT,
                ash_pulse=PULSE,
                cpu_max_abs_error=(traces - cpu).abs().max().item(),
                runs=runs,
            )
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edges", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    node = Eat4CircuitNode("ASH release")
    node.configure(edges=str(args.edges))
    workflow = WorkflowBuilder("eat4_connectome")
    workflow.add_node(node)
    if not workflow.build().execute() or not node.get_output("report"):
        raise SystemExit("Circuit simulation failed")
    args.output.write_text(json.dumps(node.get_output("report"), indent=2) + "\n")
