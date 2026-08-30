# AutoResearcher — installation and operation manual

AutoResearcher is a self-improving multi-agent system with exactly one objective:
raise **SANA-WM stage-1**'s score on the **WBench navigation split**, by changing only
the *data* the world model is trained on. It runs unattended — it picks a starting
point, edits its own code, writes a new training recipe, trains the world model,
scores it on the benchmark, and repeats.

> **This manual documents the `live-run` branch** (tag `v0.1-first-live-run`) — the
> exact state of all three repositories that the first real end-to-end run was launched
> from. Every command, flag, check and number below was verified against it. `main` has
> moved on since and works differently; do not mix instructions between the two. Step 2
> puts you on the right branch.

This document is for someone who has never seen this codebase and just needs to get it
running. It covers all three repositories (AutoResearcher, Sana, WBench), all three
conda environments, every download, and every command. Follow it top to bottom.

> **You are spending real money and real GPU-hours.** The agents run against a paid
> LLM API, and each iteration of the loop trains a 1.6B video diffusion model for
> several hours on 8 GPUs. Read [Before you start](#0-before-you-start) and
> [What a run costs](#what-a-run-costs) before launching anything.

---

## Table of contents

1. [Before you start](#0-before-you-start)
2. [Step 1 — Create the workspace](#step-1--create-the-workspace)
3. [Step 2 — Clone the three repositories](#step-2--clone-the-three-repositories)
4. [Step 3 — Build the three conda environments](#step-3--build-the-three-conda-environments)
5. [Step 4 — Download the data and the model weights](#step-4--download-the-data-and-the-model-weights)
6. [Step 5 — Configure keys and paths](#step-5--configure-keys-and-paths)
7. [Step 6 — Preflight](#step-6--preflight)
8. [Step 7 — Run it](#step-7--run-it)
9. [Step 8 — Watch it](#step-8--watch-it)
10. [Step 9 — Stop, interrupt, resume](#step-9--stop-interrupt-resume)
11. [Step 10 — Read the results](#step-10--read-the-results)
12. [Where everything lives](#where-everything-lives)
13. [Configuration reference](#configuration-reference)
14. [Troubleshooting](#troubleshooting)
15. [Starting over](#starting-over)

---

## 0. Before you start

### Hardware

| | what was used | minimum |
|---|---|---|
| GPUs | 8 × A100-80GB | 4 × 24 GB, with the small-GPU recipes (see below) |
| CPU / RAM | 64 cores, 512 GB | 16 cores, 128 GB |
| Disk | ~330 GB free before a run, on one filesystem | ~330 GB |
| OS | Linux x86-64, CUDA 12.x driver | same |

**GPUs must be exclusively yours for the whole run.** The loop launches training on
every GPU it is given and holds them for hours. Someone else's job on the same cards
does not crash the run; it makes both slower and can push training into OOM. This
version has **no preflight check for busy GPUs** — check `nvidia-smi` yourself before
every launch.

If your GPUs are smaller than 80 GB, two measured stage-1 recipes ship in the Sana
repo and the agents are told about them:

| config | peak per rank | seconds/step |
|---|---|---|
| `sana_wm_stage1_recipe_base.yaml` (8×80GB) | 32.8 GB | 114–123 |
| `sana_wm_stage1_recipe_5x24gb.yaml` | 22.4 GB | 10 |
| `sana_wm_stage1_recipe_4x24gb.yaml` | 23.5 GB | 7 |

The 24 GB recipes buy their memory by shortening the training horizon, so they train
shorter camera trajectories than the benchmark scores. They work; they score worse.

Benchmark *inference* peaks around 33 GB on a full-length case. On GPUs smaller than
that, set `AR_OFFLOAD_VAE=1` (see [Configuration reference](#configuration-reference)).

### Disk budget

Plan for this much, permanently:

| item | size |
|---|---|
| Sana training corpus + VAE latent cache | 220 GB |
| WBench data + metric model weights | 50 GB |
| SANA-WM stage-1 bundle + Gemma text encoder (HF cache) | 23 GB |
| One trained checkpoint (only ever one is kept) | 10 GB |
| Per-node evaluation artifacts | ~1–2 GB (proxy), ~25 GB cap |
| **Free headroom the kernel insists on** | **40 GB at all times** |

The loop refuses to start a node with less than 40 GB free and aborts one that eats
into it. This is not paranoia — see [Troubleshooting](#troubleshooting) for the two
occasions running out of disk silently corrupted a benchmark score instead of
crashing.

### Accounts and keys

| | needed for | required? |
|---|---|---|
| An OpenAI-compatible LLM endpoint + key | the agents themselves | **yes** |
| A Hugging Face account | downloading data and weights | yes (some repos are gated) |
| A Volcengine ARK key (Doubao) | WBench's VLM-judged metrics | optional |

The first live run used **DeepSeek** (`https://api.deepseek.com`, model
`deepseek-v4-flash-vision-exp`). Any OpenAI-compatible chat-completions endpoint that
supports tool calling will work.

Without the ARK key, 7 of the 22 WBench metrics are skipped, not removed — the run
still scores on the remaining ones, and the metrics light up automatically if you add
the key later. Cost is small (~465 calls, roughly $0.30–$2 per full evaluation). The
first live run had no ARK key, so every score quoted in this document excludes them.

### What a run costs

Measured on the first live run (5 nodes, 8 × A100-80GB, DeepSeek at
$0.44 / $1.32 per million input / output tokens):

- **LLM spend: $6.21** across all five nodes.
- **GPU time: ~29 GPU-hours.**
- One node takes **8–14 hours** end to end if its agents train for a full recipe.
- The scoring rung after each node (32 proxy cases) takes **~25–40 minutes**.

---

## Step 1 — Create the workspace

Everything lives under one directory. The layout is **not** configurable by accident —
the code derives the workspace from its own location (`kernel/config.py`:
`WORKSPACE = <this repo>/..`), so the three repos must be siblings with exactly these
names:

```
<workspace>/
├── AutoResearcher/   ← this repo (kernel + agent layer)
├── Sana/             ← the world model being trained
├── WBench/           ← the benchmark (never modified)
├── archive/          ← created on first run: all run state
└── cache/            ← created on first run: HF/torch caches, model weights
```

Pick a disk with the space from the budget above, and create it:

```bash
export WS=/path/to/AutoResearchWM        # <-- change this
mkdir -p "$WS"
cd "$WS"
```

`Sana/` and `WBench/` may be **symlinks** to checkouts elsewhere on the machine —
that is how the original deployment does it, because those two repos are large and
shared. Both forms work.

---

## Step 2 — Clone the three repositories

The GitHub repository names do **not** match the directory names the code expects.
**You must rename each one after cloning.** This is the single most common setup
mistake.

```bash
cd "$WS"

git clone git@github.com:seermer/AutoResearcher-WM-Autoresearch.git AutoResearcher
git clone git@github.com:seermer/Sana-WM-Autoresearch.git          Sana
git clone git@github.com:seermer/WBench-WM-Autoresearch.git        WBench
```

| clone from | rename to |
|---|---|
| `AutoResearcher-WM-Autoresearch` | `AutoResearcher` |
| `Sana-WM-Autoresearch` | `Sana` |
| `WBench-WM-Autoresearch` | `WBench` |

### Check out the live-run release

The default branch of these forks is `main`, which has moved on. AutoResearcher and
Sana both carry a **`live-run`** branch pinned to the tag `v0.1-first-live-run` — the
code exactly as the first live run executed it, which is what this manual describes.
Check it out in both:

```bash
git -C "$WS/AutoResearcher" checkout live-run
git -C "$WS/Sana"           checkout live-run
```

WBench has a single commit on `main` and no branches; leave it as cloned.

**Do both, and do them before `bootstrap`.** Nothing in the system pins a branch name;
`bootstrap` cuts the root node from whatever commit each repo has checked out at that
moment, taken independently for AutoResearcher and for Sana — it only ever *reads*
`live-run`/`main`, and every node it creates lives on its own new `node/<id>` branch, so
`live-run` and `main` themselves are never at risk. What *is* at risk is the archive:
checking out `live-run` in one repo and leaving the other on `main` bakes that mismatched
pair into the root, silently, and every later node inherits from it. Fixing it means
wiping `archive/` and the `node/*` branches per [Starting over](#starting-over) and
re-bootstrapping correctly — cheap if you catch it right away, costly in lost
GPU-hours and API spend the longer the tree has grown on top of it.

Confirm all three:

```bash
for r in AutoResearcher Sana WBench; do
  echo "== $r: $(git -C $r rev-parse --abbrev-ref HEAD) $(git -C $r rev-parse --short HEAD)"
done
```

You should see `live-run` for AutoResearcher and Sana (Sana at `9e2bae5`), and `main`
at `0ac3f48` for WBench. AutoResearcher's `live-run` is the tag `v0.1-first-live-run`
(`855d98b`) plus this manual; `git diff v0.1-first-live-run live-run -- . ':!README.md'`
prints nothing.

All three forks contain everything the code needs. In particular the WBench fork has
its third-party dependencies (MegaSAM, RAFT, SAM2, AMT, HPSv3, Depth-Anything-3,
TransNetV2) **vendored as ordinary files** — do not run `git submodule update`, there
is nothing to fetch.

---

## Step 3 — Build the three conda environments

Three separate environments, because their dependency stacks genuinely conflict
(different PyTorch versions, different Python versions).

| env | Python | used by | who activates it |
|---|---|---|---|
| `autoresearch` | 3.11 | the loop and the agents | **you** |
| `sana` | 3.11 | training and video generation | the kernel, as a subprocess |
| `wbench` | 3.10 | the benchmark metrics | the kernel, as a subprocess |

You only ever activate `autoresearch` yourself. The other two are invoked by absolute
path — see [Step 5](#step-5--configure-keys-and-paths), which is where you tell the
kernel where they are.

Install [Miniforge](https://github.com/conda-forge/miniforge) first if you do not have
conda. `nvcc` must be on your `PATH` and match your driver's CUDA major version
(`nvcc --version`).

### 3a. `sana` — the world model

The Sana repo ships its own installer. It creates the env, pins the CUDA toolkit,
installs the cu128 torch wheels, and builds mmcv, flash-attn and Pi3X:

```bash
cd "$WS/Sana"
bash environment_setup.sh sana
```

Takes 30–60 minutes; most of it is compiling flash-attn. Notes:

- Python 3.11 is mandatory (Triton 3.5's JIT source-inspection breaks on 3.10).
- Speed the compile up with `MAX_JOBS=16 NVCC_THREADS=4 bash environment_setup.sh sana`.
- Transformer Engine is installed best-effort at the end and is **not needed** here —
  it only enables fp8/fp4 inference. If it fails, ignore the warning, or skip it
  outright with `SANA_SKIP_TE=1 bash environment_setup.sh sana`.
- Re-running the script on an existing env is safe and reconciles versions.

Verify:

```bash
conda run -n sana python -c "import torch, sana; print(torch.__version__, torch.cuda.is_available())"
```

### 3b. `wbench` — the benchmark

WBench also ships an installer. **Pass the CUDA build explicitly** — if you omit it and
`nvcc` is not found, it silently falls back to `cu118` and the MegaSAM CUDA extensions
fail to build on a CUDA-12 machine:

```bash
cd "$WS/WBench"
bash tools/install.sh wbench cu124        # cu124 for CUDA 12.x, cu121, or cu118
```

Verify:

```bash
conda activate wbench
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python tools/verify_install.py
conda deactivate
```

Two metrics will not work in this environment, and that is the configuration the live
run used — **leave them alone**:

- `hpsv3_quality` errors on every case (a pyarrow/`datasets` version conflict). It is
  excluded from scoring at bootstrap. See
  [Troubleshooting](#hpsv3_quality-is-missing-from-the-score) before you try to fix it.
- `visual_plausibility` needs a second environment (`wbench-vp`) and another 58 GB of
  Qwen3-VL weights. AutoResearcher never scores it. Skip that install entirely.

### 3c. `autoresearch` — the loop itself

Small and pure-Python:

```bash
conda create -n autoresearch python=3.11 -y
conda activate autoresearch
cd "$WS/AutoResearcher"
pip install -r requirements.txt
```

Verify:

```bash
python -c "import langgraph, langchain_openai, openai; print('ok')"
```

### 3d. Note the paths

Write down where conda put the two subprocess environments — you need them in Step 5:

```bash
conda run -n sana   python -c "import sys; print(sys.executable)"
conda run -n wbench python -c "import sys; print(sys.executable)"
```

---

## Step 4 — Download the data and the model weights

Four downloads, ~290 GB total. Do them in any order; they are all resumable.

```bash
conda activate autoresearch
pip install -U huggingface_hub
hf auth login          # required: some repos are gated
```

**Put the caches on the same big disk as the workspace, not in `$HOME`.** The kernel
does this for its own subprocesses automatically, but a manual `hf download` obeys
whatever `HF_HOME` says, so set it for this shell:

```bash
export HF_HOME="$WS/cache/huggingface"
export HF_HUB_DISABLE_XET=1        # xet staging balloons and then fails mid-download
```

### 4a. The Sekai training corpus (~220 GB)

The world model trains on a camera-annotated Sekai-Game corpus, shipped with a
precomputed VAE latent cache. Download it **into the Sana checkout** so the relative
paths in the training configs resolve:

```bash
cd "$WS/Sana"
hf download Efficient-Large-Model/SANA-WM-example-training-dataset \
  --repo-type dataset \
  --revision 4d965e94b9ea11b9c5ba085251ffa7a0345e006f \
  --local-dir .
```

Pin that revision. When it finishes you must have exactly these two directories:

```
Sana/data/sekai_game_train_961frames_16fps_ovl640/     185 GB — clips + camera + captions + filters
Sana/data/vae_cache/LTX2VAE_diffusers_704x1280/...      35 GB — precomputed latents
```

Redistributed for non-commercial research use only; read the dataset card and its
`LICENSE`/`NOTICE.md`.

> On the first run, the kernel registers `Sana/data` as the immutable base data shard
> and **chmods it read-only** so no agent can write through to it. That is expected. If
> you ever need to re-download it, `chmod -R u+w Sana/data` first.

### 4b. WBench cases and metric weights (~50 GB)

```bash
cd "$WS/WBench"
hf download meituan-longcat/WBench --repo-type dataset --local-dir data/ --exclude "splits/*"
hf download meituan-longcat/WBench-weights --local-dir weights/
```

Afterwards `WBench/data/cases/` must hold the case definitions and `WBench/weights/`
the metric checkpoints (SAM2, DA3, MegaSAM, RAFT, HPSv3, CLIP, aesthetic, …).
`WBench/third_party/mega-sam/torchhub` is a symlink into `weights/torch_hub` that
WBench creates itself on first use — you do not need to make it.

Quick check (should print `158` and `289`):

```bash
conda activate autoresearch
cd "$WS/AutoResearcher"
python -c "from kernel.cases import navi_cases, all_cases; print(len(navi_cases()), len(all_cases()))"
```

### 4c. SANA-WM stage-1 weights and the depth model (~23 GB)

The kernel downloads these on its own before the first evaluation, but pre-fetching
lets you catch an auth problem now rather than eight hours in:

```bash
cd "$WS/AutoResearcher"
python -c "from kernel import weights; print(weights.ensure_stage1()); print(weights.ensure_metric_models())"
```

That pulls:

- `Efficient-Large-Model/SANA-WM_bidirectional` — **config + DiT + VAE only, 13 GB**
- `Efficient-Large-Model/gemma-2-2b-it` — the text encoder, 9.8 GB
- `lpiccinelli/unidepth-v2-vitl14` — MegaSAM's depth model, pinned revision

> **Never let anything resolve `SANA-WM_bidirectional` by repo id.** The full repo is
> ~96 GB, 84 GB of which is an LTX-2 refiner and a Gemma-3-12B encoder this project
> never loads. The kernel materializes a local bundle and hands out file paths
> precisely to avoid that. If you see a 96 GB download start, stop it.

The UniDepth download looks trivial and is not: WBench runs MegaSAM with one worker per
GPU, and eight workers racing a cold cache all fail their download at once. MegaSAM
then writes no camera poses and the navigation metrics report *zero applicable cases*
with no error at all. Warming it serially, once, up front is what prevents that.

---

## Step 5 — Configure keys and paths

### 5a. The LLM endpoint

Create `AutoResearcher/.env` (git-ignored):

```bash
cd "$WS/AutoResearcher"
cat > .env <<'EOF'
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.deepseek.com
EOF
chmod 600 .env
```

`.env` is loaded automatically by `kernel/config.py`. Anything in it can also be set as
a normal shell environment variable.

To enable WBench's VLM-judged metrics, add your Volcengine ARK key to the same file
(the live run did not — adding it changes which metrics get pinned, so decide before
you bootstrap):

```
VLM_API_KEY=your-ark-key
# optional, defaults shown:
# VLM_API_URL=https://ark.cn-beijing.volces.com/api/v3
# VLM_MODEL_NAME=doubao-seed-2-0-lite-260215
```

### 5b. The two subprocess interpreters — **required**

The kernel invokes the `sana` and `wbench` environments by **absolute path**, and the
built-in defaults point at the original machine's home directory
(`/home/zhantaoy/miniforge3/envs/...`). On any other machine you must override them
with the paths you noted in Step 3d:

```bash
cat >> .env <<EOF
AR_SANA_PYTHON=$(conda run -n sana python -c "import sys;print(sys.executable)")
AR_WBENCH_PYTHON=$(conda run -n wbench python -c "import sys;print(sys.executable)")
EOF
```

`cli.py doctor` in the next step fails loudly on `sana python` / `wbench python` if
these are wrong.

### 5c. GPUs and pricing (optional)

```
AR_GPUS=0,1,2,3,4,5,6,7        # default: all eight
AR_PRICE_IN=0.44               # USD per 1M input tokens — makes the monitor show dollars
AR_PRICE_OUT=1.32              # USD per 1M output tokens
```

Those two prices are the ones the live run was billed at. The full list of knobs is in
[Configuration reference](#configuration-reference). The defaults are correct for an
8×A100 host; you can run with nothing but the keys and the two interpreter paths.

---

## Step 6 — Preflight

Three commands. Run all three, every time, before spending anything. None of them uses
a GPU or costs a token.

```bash
conda activate autoresearch
cd "$WS/AutoResearcher"

python cli.py doctor                  # 19 checks, ~7 s
python scripts/selftest.py            # loop mechanics end to end, ~8 s
python scripts/monitor_selftest.py    # capture + web monitor, ~3 s
```

The two selftests must each end with `ALL CHECKS PASSED`. They clean up after
themselves: `archive/` should be untouched and no new git branches should remain.

`cli.py doctor` prints one line per check and exits non-zero if any fails:

| check | what to do when it says FAIL |
|---|---|
| `sana python` / `wbench python` | `AR_SANA_PYTHON` / `AR_WBENCH_PYTHON` are wrong → Step 5b |
| `Sana repo` / `WBench repo` | the directory is missing or misnamed → Step 2 |
| `WBench cases` | the WBench dataset download is incomplete → Step 4b |
| `base corpus` / `vae cache` | the Sekai dataset is missing or in the wrong place → Step 4a |
| `stage-1 trainer` / `recipe config` / `small-GPU recipes` | wrong Sana commit, or an incomplete clone → Step 2 |
| `LLM endpoint set` | `OPENAI_BASE_URL` is unset → Step 5a |
| `stage-1 weights bundle` | the SANA-WM bundle did not download → Step 4c |
| `caches off $HOME` | something set `HF_HOME` into your home directory → Step 4 |
| `disk headroom (>40 GB)` | free space, or point `AR_ARCHIVE_DIR` at a bigger disk |
| `online tool: …` (5 checks) | no outbound internet, or a search backend is down |

Those last five probe the search tools the agents use, each with a question whose
answer is known. They matter more than they look: a search tool that answers
"(no results)" to everything reads to an agent as *"this data does not exist"*, and it
will burn a node's entire budget concluding exactly that.

A healthy preflight ends like this:

```
  proxy cases: 32  full navi: 158
  VLM metrics: SKIPPED (no VLM_API_KEY)
  free disk: 153.6 GB   gpus: 0,1,2,3,4,5,6,7
  archive: /path/to/workspace/archive
```

Then do the one check the tool does not do — confirm nobody else is on your GPUs:

```bash
nvidia-smi
```

---

## Step 7 — Run it

### 7a. Baseline first (free)

```bash
python cli.py bootstrap
```

This evaluates the **released, untrained** SANA-WM stage-1 checkpoint on the 32-case
proxy and stores it as root node `n0000`. It takes about 25–40 minutes on 8 GPUs, uses
**zero API tokens**, and is the cheapest possible proof that your GPU path, weights,
data and benchmark install all actually work.

Expect a score near **82.8** (the live run's baseline was 82.839). You will also see,
once:

```
WARNING: no case produced these metrics, so they cannot be optimised: ['hpsv3_quality']
```

That is the metric set being pinned, and it is the expected result for this release —
see [Troubleshooting](#hpsv3_quality-is-missing-from-the-score).

### 7b. The loop

```bash
python cli.py run --max-nodes 5
```

`run` does `bootstrap` itself if the archive is empty, so you can go straight here.
Each iteration picks a promising node, lets it rewrite its own code and its data
recipe, trains the world model from the released base, scores it, and inserts the
result into the tree. Then it repeats.

Useful flags:

```bash
python cli.py run --max-nodes 20        # how many nodes to add before stopping
python cli.py run --seed 42             # deterministic node selection
```

For anything longer than a couple of nodes, detach it and keep the log:

```bash
nohup python cli.py run --max-nodes 20 > ../run.log 2>&1 &
echo $! > ../run.pid
```

The loop is self-healing. A node that crashes, times out, breaks its own code, or
produces no checkpoint is discarded ("trashed"), and the next iteration starts fresh
from a different parent. It stops on its own when it reaches `--max-nodes`, when
nothing is left worth expanding, or after 12 consecutive failures.

**What you should see in `run.log`** — one line per node, exactly as the live run
printed them:

```
[1/2] n0007 status=trash score=None best=n0001@82.884
[2/2] n0008 status=ok    score=82.903 best=n0008@82.903
```

Many nodes failing is normal and expected — six of the live run's nine did.

---

## Step 8 — Watch it

### The web monitor

```bash
python cli.py monitor          # http://127.0.0.1:8787
```

A separate, **read-only** process. Start and stop it whenever you like, including
mid-run; it cannot disturb anything. Over SSH, forward the port:

```bash
ssh -L 8787:127.0.0.1:8787 <host>      # then open http://127.0.0.1:8787 locally
```

It shows, live:

- every model call in execution order, with its full prompt and response
- every tool call with its full input and output
- the tree of nodes with scores, and which one is currently being expanded
- a per-metric comparison across nodes
- token spend, by node and by role (in dollars, if you set `AR_PRICE_IN`/`OUT`)
- GPU, disk and stray-process state
- **the generated videos, playable in the page**

To see dollars instead of raw token counts:

```bash
AR_PRICE_IN=0.44 AR_PRICE_OUT=1.32 python cli.py monitor
```

If the page says `LOOP NOT RUNNING` while the loop is clearly alive, the loop was
`kill -9`'d at some point and left no closing event. Harmless.

**Video playback needs an ffmpeg.** The generator writes MPEG-4 Part 2, which no
browser decodes, so the monitor transcodes each clip to H.264 the first time you press
play (~0.5 s) and caches it in `cache/webvideo/`. It finds ffmpeg on your `PATH`,
inside the `sana` env's `imageio-ffmpeg` wheel, or wherever `AR_FFMPEG` points. With
none of those, videos are served unconverted and stay black.

### From the terminal

```bash
python cli.py status        # the whole tree, scores, and which node is best
tail -f ../run.log          # the loop's own log
```

Per-node logs live in `archive/nodes/<id>/logs/` — that is where a training run's
output, and any agent's stdout, ends up.

---

## Step 9 — Stop, interrupt, resume

```bash
# foreground: Ctrl+C
# background: kill $(cat ../run.pid)
python cli.py stop          # <-- do not skip this
```

**Always run `cli.py stop` after interrupting.** Training is launched detached, so
killing the loop does *not* stop it: `torchrun` keeps holding every GPU indefinitely.
`stop` reaps those processes and tells you the loop's PID if it is somehow still alive.

Nothing else is needed to resume. The next `run` automatically discards the
half-finished node, prunes its worktrees, and continues from the existing tree:

```bash
python cli.py run --max-nodes 10
```

---

## Step 10 — Read the results

### The tree

```bash
python cli.py status
```

The real tree from the first live run (the last column is each node's own one-line
summary of what it changed):

```
n0000 [ok] score=82.839 cmp=0.259 clade=9 baseline: released SANA-WM stage-1 teacher, untrained
  n0001 [ok] score=82.884 (+0.05) cmp=0.296 clade=5 shift the caption mixture toward narrative
    n0002 [trash] score=None cmp=0.000 clade=1
    n0003 [trash] score=None cmp=0.000 clade=1
    n0007 [trash] score=None cmp=0.000 clade=1
    n0008 [ok] score=82.903 (+0.02) cmp=0.508 clade=1 raise the narrative proportion to 80/20
  n0004 [trash] score=None cmp=0.000 clade=1
  n0005 [trash] score=None cmp=0.000 clade=1
  n0006 [trash] score=None cmp=0.000 clade=1

nodes=9 alive=3 expandable=3 best=n0008@82.903
```

`[trash]` is a node that failed; `status` does not show why, but
`archive/nodes/<id>/node.json` records the reason in its `failure` field.

Those scores are the cheap **32-case proxy** rung, which only has to *rank* nodes
against each other.

### Promote a node to the real benchmark

The proxy is not the number you would report. To score a node on the canonical
158-case navigation split — ~3 hours of generation on 8 GPUs, at the benchmark's own
settings (4 s turns, 60 diffusion steps):

```bash
python cli.py eval --node n0008 --full
```

It always regenerates the videos. That is deliberate: reusing them would silently score
the *previous* checkpoint's output and report it as this one's.

The result is printed as JSON and stored back on the node as `full_score`.

### Raw artifacts

Everything about one node is under `archive/nodes/<id>/`:

| path | what |
|---|---|
| `node.json` | score, per-metric numbers, parent, checkpoint path, failure reason |
| `logs/` | training log, per-phase agent logs, shell output |
| `wbench_proxy/<id>/videos/` | the generated mp4s |
| `wbench_proxy/<id>/evaluation/report.json` | the benchmark's own per-metric report |
| `wbench_full/<id>/` | the same, for a `--full` promotion run |
| `trace.jsonl` | every model and tool call the node made |

The per-metric numbers are also in `node.json` under `metrics`, and the monitor's
comparison panel puts them side by side across nodes. **Read them.** See the
[`perspective_consistency` trap](#the-perspective_consistency-trap) below for why the
aggregate score alone will mislead you.

---

## Where everything lives

After a run:

```
<workspace>/
├── AutoResearcher/              this repo
│   ├── cli.py                   every command you will type
│   ├── kernel/                  the loop, evaluation, sandbox — never modified by agents
│   ├── agents/                  the agent layer — agents rewrite this
│   ├── scripts/                 the two selftests
│   └── .env                     your keys (git-ignored)
├── Sana/                        world model + the 220 GB corpus under data/
├── WBench/                      benchmark + 50 GB of metric weights
├── archive/                     ALL RUN STATE
│   ├── nodes/<id>/              per-node results, logs, videos, traces
│   ├── worktrees/<id>/          each node's private git checkouts (transient)
│   ├── current/<id>.pth         the single retained checkpoint (~10 GB)
│   ├── datastore/shards/        content-addressed, immutable training data
│   ├── traces/                  the monitor's event streams
│   └── metric_set.json          the metric set the archive scores on, pinned at bootstrap
├── cache/                       HF/torch caches, model weights, transcoded videos
└── run.log
```

Two things worth knowing about that layout:

- **Only one trained checkpoint exists at a time.** Every node trains from the released
  base, never from its parent, so an older node's weights are never needed again and
  are deleted the moment a newer one lands. `archive/current/` holds exactly one file.
- **`archive/` is the entire run.** Delete it and you have a fresh system. Keep it and
  a later `run` continues where you left off. `cache/` is separate on purpose — it is
  ~43 GB of downloads that take hours to replace.

---

## Configuration reference

Every setting is an environment variable, readable from `.env` or the shell. The
defaults are correct for the original 8×A100 host.

### Paths — set these if your layout differs

| var | default | meaning |
|---|---|---|
| `AR_SANA_PYTHON` | a hard-coded path | **the `sana` env's python — almost certainly needs setting** |
| `AR_WBENCH_PYTHON` | a hard-coded path | **the `wbench` env's python — same** |
| `AR_SANA_DIR` | `<workspace>/Sana` | the Sana checkout |
| `AR_WBENCH_DIR` | `<workspace>/WBench` | the WBench checkout |
| `AR_ARCHIVE_DIR` | `<workspace>/archive` | where all run state goes |
| `AR_CACHE_DIR` | `<workspace>/cache` | HF/torch/temp caches — keep this off `$HOME` |
| `AR_FFMPEG` | auto-detected | ffmpeg binary for the monitor's video playback |

### Keys

| var | meaning |
|---|---|
| `OPENAI_API_KEY` | the agents' LLM key |
| `OPENAI_BASE_URL` | the agents' endpoint |
| `VLM_API_KEY` | enables WBench's 7 VLM-judged metrics |

### Compute and budget

| var | default | meaning |
|---|---|---|
| `AR_GPUS` | `0,1,2,3,4,5,6,7` | GPUs used for training and generation |
| `AR_MAX_NODES` | `1000` | node cap (`--max-nodes` overrides) |
| `AR_MIN_FREE_DISK_GB` | `40` | refuse to start a node below this |
| `AR_NODE_DISK_GB` | `25` | per-node disk quota |
| `AR_EDIT_SELF_SECONDS` | `3600` | timeout for the self-editing phase |
| `AR_IMPROVE_SECONDS` | `43200` | timeout for the recipe + training phase |
| `AR_EVAL_SECONDS` | `21600` | timeout for one evaluation rung |
| `AR_TRAIN_GRACE_SECONDS` | `14400` | extra wait if training is still writing at deadline |
| `AR_MAX_CONSECUTIVE_TRASH` | `12` | halt after this many failures in a row |

### Evaluation

| var | default | meaning |
|---|---|---|
| `AR_PROXY_CASES` | `32` | size of the cheap ranking rung |
| `AR_PROXY_TURN_SECONDS` | `2.0` | seconds per turn, proxy rung |
| `AR_PROXY_STEP` | `30` | diffusion steps, proxy rung |
| `AR_FULL_STEP` | `60` | diffusion steps, full rung |
| `AR_OFFLOAD_VAE` | `0` | set to `1` on GPUs smaller than ~33 GB |

### Models and spend

| var | default | meaning |
|---|---|---|
| `AR_MODEL` | `deepseek-v4-flash-vision-exp` | model for every agent role |
| `AR_MODEL_META`, `AR_MODEL_ENGINEER`, `AR_MODEL_SCOUT`, `AR_MODEL_ANALYST` | `$AR_MODEL` | per-role override |
| `AR_LLM_EXTRA_BODY` | `{"thinking": {"type": "disabled"}}` | extra JSON fields on every request |
| `AR_LLM_TIMEOUT` | `900` | per-request timeout, seconds |
| `AR_CONTEXT_TOKENS` | `131072` | assumed context window |
| `AR_SPEND_LOG` | unset | JSONL path for per-call token usage |
| `AR_PRICE_IN` / `AR_PRICE_OUT` | unset | USD per 1M tokens, for the monitor's cost panel |

---

## Troubleshooting

Most of this system's failures are **loud** and land in `run.log` or a node's
`logs/`. The ones below are the exceptions — the ones that finish "successfully" while
being wrong. Learn these five.

### Navigation scores vanish, and nothing errors

**Symptom:** `navigation_trajectory`, `spatial_consistency` and
`gated_spatial_consistency` report *0 applicable cases*; the score looks plausible but
the whole interaction dimension is missing from it.

**Cause:** MegaSAM could not load its depth model — eight workers hit a cold
HuggingFace cache simultaneously and all failed. It writes no camera poses and moves on
without raising.

**Check:** count the pose files against the case count.

```bash
ls archive/nodes/<id>/wbench_proxy/<id>/megasam/*.npz | wc -l
```

**Fix:** warm the cache serially — `python -c "from kernel import weights; weights.ensure_metric_models()"`.
The kernel does this before every evaluation, so this should not recur.

### A benchmark run quietly loses cases to a full disk

**Symptom:** an evaluation completes and reports a normal-looking score, but for fewer
cases than you asked for.

**Cause:** MegaSAM writes ~500 MB of reconstruction scratch per case into
`WBench/third_party/mega-sam/reconstructions/` that **nothing ever reads and nothing
ever deletes**. On one long evaluation this filled the disk after 132 cases; 35
navigation cases died writing their poses, and the run still exited 0 and reported a
smaller-n score with no indication anything had gone wrong.

**Check:** always compare the case count in `report.json` against what you asked for.

**Fix:** budget ~500 MB × cases on top of everything else for a `--full` evaluation, and
run a janitor alongside it that deletes each `reconstructions/<scene>/` once the
corresponding `megasam/<scene>.npz` exists. Recovery is cheap — MegaSAM skips cases
whose `.npz` already exists, so you can backfill just the missing ones and re-run
`main.py --phase gpu --metrics navigation_trajectory,spatial_consistency` then
`--phase report`.

### Training fills the disk mid-run

**Symptom:** a node dies hours in, out of disk, with nothing in its own log explaining
why.

**Cause:** every checkpoint save writes a 9.9 GB merged `.pth` **plus** a 30 GB sharded
FSDP directory — ~40 GB per save — and Sana's `checkpoint_total_limit` prunes neither
on the FSDP path. A node choosing to save every 20 steps of a 400-step run would leave
~800 GB behind.

**Fix:** already handled — the kernel runs a disk guard for the whole training phase
that deletes sharded state once its merged checkpoint exists, and keeps only the two
newest `.pth` files. Nothing here ever resumes from a shard. If you see this anyway,
you are below the 40 GB headroom the guard needs; free space.

### `hpsv3_quality` is missing from the score

**Expected, and correct for this release.** In the `wbench` environment that
`tools/install.sh` builds, `hpsv3_quality` fails on every case: HPSv3 pulls in
`datasets` 2.14, which subclasses a pyarrow API removed after pyarrow 20. The archive
pins its metric set at bootstrap over whatever produced a number, so `hpsv3_quality` is
excluded and every node stays comparable.

It *is* fixable (`pip install --no-deps 'pyarrow==20.0.0'` plus `pip install rich` in
the `wbench` env), but **do not fix it if you want scores comparable to the live run** —
bootstrapping with it working pins it into the metric set and shifts every number.
Whatever you choose, choose it before `bootstrap` and never change it mid-run. The set
lives in `archive/metric_set.json`, and a node missing any pinned metric is failed
rather than scored, on purpose: otherwise a node could "improve" simply because an
inconvenient metric crashed.

### The `perspective_consistency` trap

The one lineage in the first live run that raised the aggregate score did so while
**dropping `perspective_consistency` by 4.18 points** — camera-pose stability across
turns, arguably the metric that matters most for a camera-conditioned world model:

| metric | n0000 (base) | n0008 |
|---|---|---|
| navigation_trajectory | 78.79 | 79.50 |
| navigation_accuracy | 65.41 | 66.19 |
| navigation_consistency | 92.18 | 92.80 |
| **perspective_consistency** | **75.13** | **70.95** |
| **aggregate score** | 82.839 | **82.903** |

The aggregate rose anyway, because small gains across three metrics outweighed one
large loss. Nothing in the loop notices this. **Read the per-metric table before
believing a score went up for a good reason** — `metrics` in `node.json`, or the
comparison panel in the monitor.

### Other common problems

| symptom | cause and fix |
|---|---|
| Videos are black in the monitor | no ffmpeg found → install one, or set `AR_FFMPEG` |
| Training is slow, or OOMs at a batch size that used to fit | another user's job on your GPUs. There is no preflight for this in this release → check `nvidia-smi` before launching |
| A 96 GB download starts | something resolved `SANA-WM_bidirectional` by repo id → kill it, use `weights.ensure_stage1()` |
| GPUs stay busy after Ctrl+C | you skipped `python cli.py stop` → run it |
| Downloads fail with no space, but the workspace disk is fine | `HF_HOME` points into `$HOME` → Step 4 |
| Every node fails immediately | check `archive/nodes/<id>/logs/*.log`; usually a bad `AR_SANA_PYTHON`/`AR_WBENCH_PYTHON` |
| Half the nodes fail | normal — `node.json`'s `failure` field records each one's reason |

---

## Starting over

To discard a run completely and start from nothing:

```bash
cd "$WS/AutoResearcher"
python cli.py stop
cd "$WS"
chmod -R u+w archive && rm -rf archive && mkdir archive
for r in AutoResearcher Sana; do
  git -C $r worktree prune
  git -C $r for-each-ref --format='%(refname:short)' refs/heads \
    | grep -Ev '^(main|live-run)$' | xargs -r git -C $r branch -D
done
```

That removes every node, every score, every checkpoint and every per-node git branch.
Your `live-run` checkout is untouched.

**Leave `cache/` alone.** It holds the SANA-WM bundle and the depth model — about 43 GB
that takes hours to re-download and that nothing in a run ever invalidates.

Then go back to [Step 6](#step-6--preflight).

---

## Further reading

- `DIFFICULTIES.md` (in the workspace root) — what the first live run actually found,
  written up as research obstacles rather than bug reports.
- WBench's own `README.md` — the benchmark, its 22 metrics and its leaderboard.
- `Sana/docs/sana_wm.md` — the world model and its stage-1 training recipes.
