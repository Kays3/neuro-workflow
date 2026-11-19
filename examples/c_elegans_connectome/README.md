# C. elegans connectome smoke test

This single-node NeuroWorkflow example loads the released ConnectomeGPT-Worm RealIO language-model checkpoint (seed 1337, training step 10,000). It reconstructs the fixed wiring matrix from the OpenWorm hermaphrodite edge list and checks the normalized weights entry by entry. The graph includes muscle and other non-neuronal targets.

The test uses the model definitions from the author's notebook and loads the checkpoint strictly. It supplies two GPT-2 token IDs, checks the next-token logits and loss, and repeats inference with the wiring matrix set to zero. The change in logits measures whether this forward pass depends on the matrix. Biological prediction accuracy and the benefit of the anatomical wiring require separate evaluations.

## Reproduce

Run from this directory with curl and uv. The checkpoint download is 341 MB. The first run also downloads the Python dependencies.

```sh
worm_cache="${XDG_CACHE_HOME:-$HOME/.cache}/connectome-gpt-worm/a00753a"
mkdir -p "$worm_cache"
release=https://huggingface.co/drmylesgarveylabs/connectome-gpt-worm/resolve/a00753aeb87c0bee70b57c81f071b003369138df
curl -fL --retry 2 "$release/experiment_pipelines/experiment_000/01_worm_experiments_final_sheet_%28runnable%29%20%281%29.ipynb" -o "$worm_cache/model.ipynb"
curl -fL --retry 2 "$release/checkpoints/exp1_real_io_seed1337_step010000.pt" -o "$worm_cache/checkpoint.pt"
curl -fL --retry 2 https://raw.githubusercontent.com/openworm/CElegansNeuroML/1ca08b14f4e795a973ce447565e0c97b897ee4dc/herm_full_edgelist.csv -o "$worm_cache/edges.csv"
uv run --no-project --python 3.13 --with torch==2.13.0 --with numpy==2.4.6 --with ../.. python workflow.py "$worm_cache"
```

The script verifies the SHA-256 of each input and prints JSON after all checks pass. The recorded run is in [result.json](result.json).
