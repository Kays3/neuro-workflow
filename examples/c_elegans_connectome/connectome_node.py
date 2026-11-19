"""CPU inference check for the published ConnectomeGPT-Worm RealIO checkpoint."""

import ast
import csv
import hashlib
import json
from pathlib import Path

import torch
from neuroworkflow import (
    MethodDefinition,
    Node,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
    PortType,
)
from torch import nn
from torch.nn import functional as F

REVISION = "a00753aeb87c0bee70b57c81f071b003369138df"
NOTEBOOK_HASH = "54180bbd45026b1667553fe0989008ea432b2b939fcb99de15803df766c8d7ce"
CHECKPOINT_HASH = "e027f259cd76c4f6be4fd42b8a2c4d75a13ba0cb210d556908714ba1c78c6432"


def verify(path, expected):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Unexpected SHA-256 for {path}")


def model_class(notebook):
    # Load only the reviewed definitions from the pinned upstream model cell.
    names = {
        "CausalSelfAttention",
        "TransformerBlock",
        "_init_weights",
        "ConnectomeGPT_RealIO",
    }
    cells = json.loads(notebook.read_text())["cells"]
    (source,) = (
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code"
        and "class ConnectomeGPT_RealIO(" in "".join(cell["source"])
    )
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(tree, str(notebook), "exec"), namespace)
    return namespace["ConnectomeGPT_RealIO"]


@torch.inference_mode()
def run(notebook, checkpoint, edges):
    verify(notebook, NOTEBOOK_HASH)
    verify(checkpoint, CHECKPOINT_HASH)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved["model"]
    verify(edges, "142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca")
    with edges.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    nodes = sorted({row[key].strip() for row in rows for key in ("Source", "Target")})
    index = {name: i for i, name in enumerate(nodes)}
    raw = torch.zeros((len(nodes), len(nodes)), dtype=torch.float64)
    for row in rows:
        raw[index[row["Target"].strip()], index[row["Source"].strip()]] += float(
            row["Weight"]
        )
    raw.fill_diagonal_(0)
    radius = torch.linalg.eigvals(raw).abs().max()
    torch.testing.assert_close(state["A"], (raw / radius).float(), rtol=1e-5, atol=1e-8)
    vocab_size, width = state["token_emb.weight"].shape
    model = model_class(notebook)(
        vocab_size,
        width,
        4,
        1024,
        1024,
        state["A"].to_sparse(),
        state["input_idx"],
        state["output_idx"],
    ).eval()
    model.load_state_dict(state, strict=True)
    # GPT-2 token IDs for "Hello world!". Supply the next token at each position.
    tokens = torch.tensor([[15496, 995, 0]])
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    result = model(inputs, labels=targets)
    logits = result["logits"]
    if (
        logits.shape != (1, 2, vocab_size)
        or not logits.isfinite().all()
        or not result["loss"].isfinite()
    ):
        raise RuntimeError("Invalid forward-pass output")
    nonzeros = model.A.count_nonzero().item()
    model.A.zero_()
    delta = (logits - model(inputs)["logits"]).abs().max()
    if not delta.isfinite() or delta <= 1e-5:
        raise RuntimeError(
            "Zeroing the wiring matrix did not measurably change the logits"
        )
    return {
        "status": "passed",
        "revision": REVISION,
        "checkpoint_sha256": CHECKPOINT_HASH,
        "step": saved["step"],
        "device": str(logits.device),
        "torch_version": str(torch.__version__),
        "edge_list_rows": len(rows),
        "matrix_matches_openworm": True,
        "matrix_shape": list(model.A.shape),
        "nonzero_matrix_entries": nonzeros,
        "input_nodes": model.n_input,
        "output_nodes": model.n_output,
        "token_ids": tokens.tolist(),
        "logits_shape": list(logits.shape),
        "next_token_cross_entropy": result["loss"].item(),
        "zero_matrix_max_logit_change": delta.item(),
    }


class ConnectomeSmokeNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="c_elegans_connectome_smoke",
        stage="analysis",
        tool="PyTorch",
        model_source="https://huggingface.co/drmylesgarveylabs/connectome-gpt-worm",
        description="Verify the anatomical wiring and run a pretrained RealIO language model on CPU.",
        parameters={
            "asset_dir": ParameterDefinition(
                default_value=".",
                description="Directory containing model.ipynb, checkpoint.pt and edges.csv.",
            ),
        },
        inputs={},
        outputs={
            "report": PortDefinition(
                type=PortType.DICT,
                description="Checkpoint identity, wiring dimensions, token loss and zero-matrix control.",
            ),
        },
        methods={
            "evaluate": MethodDefinition(
                description="Check the three input hashes, reconstruct the matrix and run inference.",
                inputs=[],
                outputs=["report"],
            ),
        },
    )

    def __init__(self, name):
        super().__init__(name)
        self.add_process_step("evaluate", self.evaluate, method_key="evaluate")

    def evaluate(self):
        assets = Path(self._parameters["asset_dir"]).expanduser()
        report = run(
            assets / "model.ipynb", assets / "checkpoint.pt", assets / "edges.csv"
        )
        return {"report": report}
