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
   train, return the merged checkpoint. Then the kernel runs WBench. Failure → trash.
4. **Backpropagate** — push the result up through every ancestor.
5. **Insert** — the child becomes a node; repeat.

## Fitting 24 GB GPUs

Measured per-process resident memory on A100-80GB
(`nvidia-smi --query-compute-apps`), 4 ranks unless noted:

| config | peak/rank | s/step |
|---|---|---|
| CP=4, 25 latent frames | 32.8 GB | 10 |
| CP=4, 9 latent frames | 31.8 GB | — |
| CP=1, 25 latent frames | 27.9 GB | 19 |
| CP=1, 9 latent frames | 23.5 GB | 7 |
| CP=1, 13 latent frames, **5 ranks** | **22.4 GB** | 10 |
| CP=1, 9 latent frames, 5 ranks | 21.3 GB | 7 |

Two non-obvious things drive this.

**Context parallel is the expensive part, not the sequence.** The fused GDN CP Triton
kernels allocate ~5 GB/rank *outside* PyTorch's allocator — visible only in the OOM
message as `28.81 GiB in use, 16.00 GiB allowed, 15.42 GiB allocated by PyTorch`. And
`triton_block_fusion: true` is mandatory whenever CP is on, so CP's cost cannot be
separated from CP; the only way to avoid it is `cp_size: 1`.

**`latent_frames` only becomes a lever once CP is off.** Under CP each rank holds 1/N of
the sequence, so cutting frames 64% saved 1.0 GB. Without CP each rank holds all of it
and the same cut saves 4.4 GB.

Beware measuring this with `torch.cuda.set_per_process_memory_fraction`: it caps only the
caching allocator, so a run can sit "within" a 24 GB cap while the process actually holds
33 GB. `scripts/memcap/` caps *and* reports `max_memory_reserved`, but the number that
matters is per-process resident memory.

The cost is horizon: 13 latent frames is ~6s at 16 fps against WBench cases of ~16s, so
these recipes train shorter-horizon camera control than they are scored on.

## Node isolation

Finishing a node freezes it. A child can read everything its parent had and change
none of it.

| asset | how a child gets it | why the parent is safe |
|---|---|---|
| agent code | git worktree on a new branch off the parent's | separate checkout; parent's branch is never checked out for writing |
| WM codebase | same, in the Sana repo | same |
| training data | symlink farm into `datastore/shards/` | shards are content-addressed and `chmod a-w`; writes through a link fail |
| new data | node-private `data/staging/` | sealed into the store only on success, then immutable too |
| checkpoint | trained from base, never from the parent | nothing to share |

`Node.shards` is the data manifest — a list of shard ids. A child starts from a copy of
its parent's list and appends whatever it sealed, so lineage is recorded without copying
a 220 GB corpus. The base corpus is registered once as `base-sekai`, by reference.

## Crash safety

Self-editing is the dangerous part: an agent rewriting the package its own interpreter
loaded can pull in a half-written module or a truncated prompt.

- `edit_self` **runs from a frozen detached worktree at the parent's commit** and
  **writes to a separate draft worktree** on the child's branch. The code being executed
  and the code being edited are different directories on disk, so a partial write can
  never reach the running process. The frozen snapshot is released once edit_self returns.
- `improve_recipe` then runs from the child's committed code, which nothing writes to
  during that phase (its writable roots are the Sana worktree, staging, and the out dir).
- Agents run in a **separate process** (`kernel/runners/agent_host.py`) with a timeout and
  their own process group, so a hang or a segfault is contained.
- `Loop.step()` catches everything below `SecurityError`: the node is trashed with the
  reason recorded and the loop continues. Five consecutive kernel-level crashes halt it.
- `Loop.recover()` runs before every `run()`: nodes left `PENDING` by a killed process are
  trashed, their worktrees removed, their partial checkpoints reclaimed, and both repos
  pruned of stale worktree records.

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
- config: `Sana/configs/sana_wm/stage1/sana_wm_stage1_recipe_base.yaml` (8x80GB)
- small GPUs: `..._recipe_5x24gb.yaml` (22.4 GB/rank) and `..._recipe_4x24gb.yaml`
  (23.5 GB/rank), measured per-process; see **Fitting 24 GB GPUs** below
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
