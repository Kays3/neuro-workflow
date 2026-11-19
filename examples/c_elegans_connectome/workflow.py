"""Execute the C. elegans smoke test through the NeuroWorkflow Python API."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from connectome_node import ConnectomeSmokeNode
from neuroworkflow import WorkflowBuilder


def run_workflow(asset_dir):
    node = ConnectomeSmokeNode("ConnectomeGPT")
    node.configure(asset_dir=str(asset_dir))
    builder = WorkflowBuilder("C_elegans_connectome_smoke")
    builder.add_node(node)
    if not builder.build().execute():
        raise RuntimeError("ConnectomeGPT workflow failed")
    report = node.get_output("report")
    if not report or report.get("status") != "passed":
        raise RuntimeError("ConnectomeGPT node did not return a passing report")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_dir", type=Path)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        report = run_workflow(args.asset_dir)
    print(json.dumps(report, indent=2))
