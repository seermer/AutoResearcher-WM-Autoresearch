# AutoResearcher

A recursively self-improving multi-agent system whose only objective is to raise
**SANA-WM stage-1**'s score on the **WBench navigation split**, purely by changing
the *data* the world model is trained on.

Architecture follows HyperAgent (clade-metaproductivity selection) and HGM
(self-editing agents in a growing archive).

## Layers

### Kernel — `kernel/`, not editable by agents
Holds no meta agent. Only mechanism:

| module | role |
|---|---|
| `archive.py` | tree of agent nodes, one directory + git branch pair per node |
| `cmp.py` | clade-metaproductivity, MCTS-style backup through every ancestor |
| `selection.py` | Thompson sampling over Beta posteriors on each clade |
| `vcs.py` | a git branch + worktree per node, in both repos; trash namespace |
| `security.py` | writable-root enforcement, protected trees, disk quota, process-group timeouts |
| `contract.py` | verifies every self-edit: valid diff, entry points, compiles |
| `evaluate.py` | the WBench protocol; agents can read results, never run or change it |
| `cases.py` | the fixed, stratified proxy case set |
| `tools/` | the core tools only: files, online search, shell |
| `loop.py` | the outer loop |
| `runners/` | dependency-light scripts run in the `sana` env or in an isolated agent process |

### Agent layer — `agents/`, fully editable, versioned per node
`graph.py` (the explicit LangGraph topology), `roles.py`, `prompts/*.md`,
`tools_ext/` (starts empty, the meta agent fills it), `memory.py`.
The meta agent is itself part of this layer, so it can rewrite its own prompt,
its own graph, and the tools every role sees.

`agents/entrypoints.py` is the **contract**: it must always export
`edit_self(ctx)` and `improve_recipe(ctx)`, each taking one positional argument.
This is re-verified programmatically after every self-edit.

## The loop

1. **Select** — Thompson-sample a node by its clade posterior
   `Beta(1 + Σwx, 1 + Σw(1-x))`, where a descendant `d` levels below contributes
   weight `0.9^d` and normalized score `sigmoid((score - baseline)/2)`. High-CMP
   clades get exploited; thin clades keep a wide posterior and get explored.
2. **Self-modify** — run the selected node's own `edit_self` in an isolated
   process, with its agent worktree first on `sys.path`. Verify diff, entry
   points and compilation. Failure → trash branch.
3. **Evaluate** — run the *edited* agent's `improve_recipe`: change the data,
   train a LoRA, return the adapter. Then the kernel runs WBench. Failure → trash.
4. **Backpropagate** — push the result up through every ancestor.
5. **Insert** — the child becomes a node; repeat.

## What agents may and may not touch

| | |
|---|---|
| editable | `agents/**`, the Sana training path, configs, the shared datastore |
| **forbidden** | all of `WBench/`, `kernel/`, `Sana/inference_video_scripts/` |

Sana's inference path is protected on purpose: if agents could change how the
model is sampled, "improvement is data-driven" would not be enforceable.

## Evaluation

Two rungs. The proxy only has to *rank* nodes, so it is deliberately cheaper; the
full eval is the promotion gate and uses canonical settings.

| | cases | seconds/turn | DiT steps | generation on 8xA100 |
|---|---|---|---|---|
| proxy (every node) | 32, stratified | 2.0 | 30 | ~10 min |
| full (`eval --full`) | 158 | 4.0 | 60 | ~3 h |

Measured: one 6-turn case at 573 frames / 60 steps takes 804 s on a single A100.
Cost scales roughly with frames x steps, so the proxy settings cut it ~4x per case.
WBench's own metric phases (SAM2 + DA3 + MegaSAM precompute, then GPU metrics) run
on top of that.

- The 32 proxy cases are stratified over all 12 (scene category × perspective)
  strata and deterministic given `(n, seed)`.
- Generation is kernel-owned (`runners/wbench_generate.py`), so every node is
  measured through an identical protocol — only weights and data differ. Poses
  come from WBench's own `case_to_poses`, are slerp-interpolated to one pose per
  RGB frame, and the mp4 is written with WBench's writer settings.
- Score = leaderboard convention: mean over the dimensions that have any metric
  present. Without a VLM key that is quality (6), consistency (8) and
  `navigation_trajectory`. VLM metrics are skipped, not removed — set
  `VLM_API_KEY` and they light up automatically.

## Checkpoints

**Every node trains from the released base.** Checkpoints are never chained, so the
archive stores exactly two sets of weights at any time: the immutable base bundle and
the single newest node's checkpoint. Everything else is deleted the moment a newer one
lands — nothing older is ever needed again, and disk here has no room to spare.

- trainer: `Sana/train_video_scripts/train_sana_wm_stage1.py` (stock, unmodified)
- config: `Sana/configs/sana_wm/stage1/sana_wm_stage1_recipe_base.yaml`
- retention: `kernel/loop.py:Loop._retain` → `archive/current/<node>.pth`

FSDP2 already writes a merged, inference-loadable `.pth` next to the sharded state;
the kernel keeps that and discards the shards, which only matter for resuming.
Measured: merged checkpoint **9.9 GB**, shard directory **30 GB** (transient, deleted
after evaluation). Budget ~40 GB of headroom for the moment a save lands.

What a node inherits from its parent is its **agent code and data recipe**, not weights.
Training step count is chosen by the agents, scaled to how much data they added.

## Usage

```bash
conda activate autoresearch
python cli.py doctor                 # preflight
python scripts/selftest.py           # loop mechanics, stubbed agents + eval
python cli.py bootstrap              # build and score the baseline root node
python cli.py run --max-nodes 20     # grow the tree
python cli.py status                 # the tree, scores, CMP
python cli.py eval --node n0007 --full   # promote a node to the full 158 cases
```

## Configuration

Everything is env-overridable; see `kernel/config.py`.

| var | meaning |
|---|---|
| `AR_ARCHIVE_DIR` | where nodes, worktrees and the datastore live |
| `AR_CACHE_DIR` | HF/torch/temp caches — kept off `$HOME`, which is full on this host |
| `AR_GPUS` | GPU list for generation and training |
| `AR_PROXY_CASES` | proxy rung size |
| `AR_MODEL_META` etc. | per-role model, so a strong model can drive the meta agent |
| `VLM_API_KEY` | enables WBench's VLM metrics |
