# mirendil-test Conversation History

Generated: 2026-06-13

Note: this conversation was compacted once before this file was created. The
earlier portion is reconstructed from the retained conversation summary, not
from a verbatim transcript. The later portion is recorded from the available
turn history.

## Original Goal

The user is studying and trying to reproduce results from:

https://arxiv.org/pdf/2604.26294

The conversation focused on understanding tensor parallelism, sequence
parallelism, folded tensor-sequence parallelism (TSP), attention shapes,
communication primitives, and adding profiling/instrumentation to local
implementations.

## Paper Context

Relevant extracted paper details:

- Paper PDF was downloaded locally earlier to `/tmp/tsp2604.26294.pdf`.
- The main implementation target was Algorithm 1: TSP MHA forward pass.
- Important notation:
  - `D`: TSP parallel degree / number of ranks.
  - `p`: current sequence-shard rank.
  - `r`: weight-owner index.
  - `B`: batch size.
  - `S`: full sequence length.
  - `h` or `H`: hidden size.
  - `Sloc = S/D`.
- Paper reference setup discussed:
  - `B = 1`
  - `D = 8`
  - `h = 4096`
  - `L = 32`
  - `num_heads = 32`
  - `head_dim = 128`
  - `GQA ratio g = 1`
  - `FFN expansion F = 4`

Core TSP attention algorithm derived during the teaching sessions:

```text
Y_p = 0
for r in 0..D-1:
    broadcast [Wq_r, Wk_r, Wv_r, Wo_r]
    Q_rp = X_p @ Wq_r^T
    K_rp = X_p @ Wk_r^T
    V_rp = X_p @ Wv_r^T
    K_r, V_r = all_gather(K_rp, V_rp, over sequence ranks p)
    A_rp = causal_attn(Q_rp, K_r, V_r)
    Y_p += A_rp @ Wo_r^T
```

Key correction: TSP attention does not need a final all-reduce in the forward
output path because each rank owns different output tokens. The sum over `r`
happens locally as each weight/head shard is processed.

## Teaching Sessions Completed

The user used the `$teach` skill repeatedly. Each session created a checklist
under `sessions/teaching/`.

### Tensor Parallelism

File:

```text
sessions/teaching/2026-06-12-tensor-parallelism.md
```

Main concepts confirmed:

- Plain TP attention keeps `X` replicated across TP ranks.
- Heads/features are sharded.
- Column-parallel `Wq/Wk/Wv` produces local `Q/K/V` shaped `[B, S, H/TP]`.
- Local attention output is `[B, S, H/TP]`.
- Row-parallel `Wo_i` maps local head-space features into partial residual
  vectors `[B, S, H]`.
- `all_reduce(sum)` combines row-parallel `Wo` partial outputs in plain TP.
- `all_gather` concatenates distinct shards; it is not the right primitive for
  summed row-parallel partials.
- `broadcast` is one-to-all.
- TP reduces weight, gradient, optimizer, and projection compute roughly by TP
  degree, but input activations are replicated.
- TSP adds sequence sharding, which forces K/V exchange because queries need
  global K/V.

### Wo Output Projection in Attention

File:

```text
sessions/teaching/2026-06-12-wo-output-projection-attention.md
```

Main concepts confirmed:

- Before `Wo`, attention output is per-head/head-space features, not final
  residual updates.
- Full nonparallel attention has `A: [B, S, H]`, `Wo: [H, H]`, and
  `A @ Wo: [B, S, H]`.
- Example: `H=64`, `num_heads=4`, `head_dim=16`.
- Per-head outputs concatenate to `[B, S, 64]`.
- Row-block view:

```text
A = [A0 | A1 | A2 | A3], Ai: [B, S, 16]
Wo = [Wo0; Wo1; Wo2; Wo3], Woi: [16, 64]
A @ Wo = A0 @ Wo0 + A1 @ Wo1 + A2 @ Wo2 + A3 @ Wo3
```

- In 4-way TP, `A_i: [B, S, H/4]`, `Wo_i: [H/4, H]`,
  `A_i @ Wo_i: [B, S, H]`.
- `all_reduce(sum)` combines these partial residual contributions.
- The user's main gap was confusing head-sharded attention-space features with
  residual-space features.

### Row-Parallel vs Column-Parallel

File:

```text
sessions/teaching/2026-06-12-row-parallel-vs-column-parallel-attention-orientation.md
```

Main concepts confirmed:

- Under row-vector convention, `Y = X @ W`.
- `X: [B, S, H]`, `W: [H, O]`, `Y: [B, S, O]`.
- Rows of `W` correspond to input features.
- Columns of `W` correspond to output features.
- Column-parallel splits output columns:

```text
W_i: [H, O/TP]
X @ W_i -> [B, S, O/TP]
```

- Row-parallel splits input rows:

```text
X_i: [B, S, H/TP]
W_i: [H/TP, O]
X_i @ W_i -> partial [B, S, O]
```

- Row-parallel outputs must be summed across ranks.
- Attention uses column-parallel Q/K/V followed by row-parallel `Wo`.
- Better mental model: reason from input/output axes and shapes, not visual
  "rows" and "columns" on arbitrary tensors.

### Sequence Parallelism and TSP Tradeoffs

File:

```text
sessions/teaching/2026-06-12-sequence-parallelism-tsp-tradeoffs.md
```

Main concepts confirmed:

- Plain TP keeps sequence activations replicated.
- SP shards the token axis:

```text
X_i: [B, S/D, H]
```

- SP reduces activation memory and attention work per rank.
- SP keeps model weights, gradients, and optimizer states replicated.
- Attention is hard under SP because local Q needs remote K/V.
- SP attention recovers global context through K/V exchange.
- Contiguous causal sequence splits can be imbalanced because later tokens
  attend to longer prefixes.
- Zigzag/striped partitioning balances causal attention load.
- TSP folds TP and SP onto one device axis.
- In TSP attention, ranks iterate over broadcast attention weight shards and
  all-gather K/V for the current shard.
- In TSP MLP, weight shards rotate while each rank accumulates outputs for
  local tokens.
- TSP reduces both model-state and activation memory by roughly the folded
  degree, but adds runtime weight movement.

### TP+SP vs TSP Tradeoffs

File:

```text
sessions/teaching/2026-06-12-tp-sp-vs-tsp-tradeoffs.md
```

Main concepts confirmed:

- TP+SP uses two mesh axes:
  - `T`: tensor parallel degree.
  - `Sigma`: sequence parallel degree.
- Fixed total devices `D = T * Sigma` splits the budget between model-state
  savings and activation-memory savings.
- TP+SP weight-state memory scales as `1/T`.
- TP+SP activation memory scales as `1/Sigma`.
- TSP folds both reductions onto one axis of size `D`, aiming for both memory
  classes to scale roughly as `1/D`.
- TSP is not strictly better; it adds weight movement and depends on topology
  and overlap.
- TSP is attractive when memory is the bottleneck, context/batch is large, and
  extra weight traffic stays on fast links or overlaps with compute.
- TP+SP is attractive when independent control over `T` and `Sigma` is useful,
  or when TSP weight movement cannot be hidden.

### Implementation of TSP Attention

File:

```text
sessions/teaching/2026-06-12-tsp-attention-implementation.md
```

Completed checklist:

```text
Progress: 18/18 concepts confirmed
```

Important corrections made:

- Include `Wo` in the weight shard broadcast.
- `r` is the weight-owner index.
- `p` is the current sequence-shard rank.
- For fixed `r`, all-gather `K_rp,V_rp` over sequence ranks `p`.
- No all-reduce over `A`.
- Use `causal_attn(Q, K, V)`, not `causal_attn(Q, K, K)`.
- Final TSP output accumulation:

```text
Y_p += A_rp @ Wo_r
```

or, depending on weight orientation:

```text
Y_p += A_rp @ Wo_r^T
```

## Shape Overview Discussed After Compaction

The user asked for shape reminders. The assistant explained:

```text
D = number of ranks/devices in the TSP group
p = current sequence-shard rank
r = current weight-shard owner being broadcast
H = model hidden size / residual width / embedding dimension
```

TSP attention shape flow:

```text
X_p: [B, S/D, H]

Wq_r, Wk_r, Wv_r: [H/D, H]
Wo_r:              [H, H/D]

Q_rp = X_p @ Wq_r^T -> [B, S/D, H/D]
K_rp = X_p @ Wk_r^T -> [B, S/D, H/D]
V_rp = X_p @ Wv_r^T -> [B, S/D, H/D]

K_r = all_gather(K_rp) -> [B, S, H/D]
V_r = all_gather(V_rp) -> [B, S, H/D]

A_rp = causal_attn(Q_rp, K_r, V_r) -> [B, S/D, H/D]

Y_rp = A_rp @ Wo_r^T -> [B, S/D, H]
Y_p = sum_r Y_rp     -> [B, S/D, H]
```

Sequence parallel attention flow:

```text
X_p: [B, S/D, H]

Q_p = X_p @ Wq        # [B, S/D, H]
K_p = X_p @ Wk        # [B, S/D, H]
V_p = X_p @ Wv        # [B, S/D, H]

K = all_gather(K_p)   # [B, S, H]
V = all_gather(V_p)   # [B, S, H]

A_p = causal_attn(Q_p, K, V)  # [B, S/D, H]

Y_p = A_p @ Wo        # [B, S/D, H]
```

Clarifications:

- `X_p` in SP is neither row-parallel nor column-parallel. It is
  sequence-sharded.
- Row-parallel and column-parallel terms are mostly relevant to TP linear
  layers because they describe sharding axes of a 2D weight matrix.
- For activations, it is clearer to say batch-sharded, sequence-sharded,
  hidden-sharded, head-sharded, or feature-sharded.

## CLAIMS.md

The user asked for shape claims to validate from the paper. The assistant
created:

```text
CLAIMS.md
```

It contains:

- TSP attention shape checks.
- Head bucketing shape checks.
- Plain SP attention shape checks.
- Memory scaling claims for TP, SP, TSP, TP+SP.
- Minimal instrumentation checklist.

Important tested paper shape noted:

```text
B = 1
D = 8
H = 4096
num_heads = 32
head_dim = 128
S = 8192, 65536, and measured ranges such as 16K..128K
```

For `S=8192`, `D=8`:

```text
X_p:              [1, 1024, 4096]
Q/K/V local:      [1, 1024, 512]
K/V after gather: [1, 8192, 512]
Y_p:              [1, 1024, 4096]
```

For `S=65536`, `D=8`:

```text
X_p:              [1, 8192, 4096]
Q/K/V local:      [1, 8192, 512]
K/V after gather: [1, 65536, 512]
Y_p:              [1, 8192, 4096]
```

## Profiling Discussion

The user asked whether latency and memory usage should be measured. The
assistant confirmed:

- Measure peak per-GPU memory.
- Measure latency / throughput.
- For memory:
  - peak allocated GPU memory per rank.
  - model-state memory.
  - activation memory.
  - temporary communication buffers.
- For latency:
  - attention forward latency.
  - MLP forward latency.
  - full block latency.
  - forward + backward if training.
  - tokens/sec or samples/sec.
- For TSP, the tradeoff is memory savings versus extra communication:
  - weight broadcasts.
  - K/V all-gathers.

The user asked how to use PyTorch profiler. The assistant suggested:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
) as prof:
    ...

torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
prof.export_memory_timeline("memory.html", device="cuda:0")
```

For `torchrun`, each rank should create its own profiler and write rank-specific
trace files.

## Codebase Files

Important files in the workspace:

```text
CLAIMS.md
iterate.py
profiler_utils.py
sequence_parallelism.py
tensor_parallelism.py
ts_parallelism.py
```

There are teaching checklist files under:

```text
sessions/teaching/
```

## Profiling Implementation Added

The assistant added `profiler_utils.py` with `profile_rank_region(...)`.

Purpose:

- synchronize ranks.
- reset CUDA peak memory stats.
- run `torch.profiler.profile(...)`.
- export Chrome trace JSON.
- export memory timeline HTML.
- print per-rank peak memory and max peak across ranks.

Core behavior:

```text
trace_dir = trace_dir or os.environ.get("TRACE_DIR", "traces")

trace={trace_dir}/{label}_rank_{rank}_trace.json
memory={trace_dir}/{label}_rank_{rank}_memory.html
```

The assistant wrapped attention calls in:

- `tensor_parallelism.py` with label `tp_attention`.
- `sequence_parallelism.py` with label `sp_attention`.
- `ts_parallelism.py` with label `tsp_attention`.

The Modal image in `iterate.py` was updated to include:

```python
.add_local_file("profiler_utils.py", remote_path="/root/profiler_utils.py")
```

## Modal Volume Work

The user asked to add a Modal Volume and collect traces.

`iterate.py` was updated to include:

```python
trace_volume = modal.Volume.from_name("tp-attn-traces", create_if_missing=True)
TRACE_ROOT = "/traces"
```

Each Modal function mounts:

```python
volumes={TRACE_ROOT: trace_volume}
```

Each run sets:

```text
TRACE_DIR=/traces/{tp,sp,tsp}
```

and commits:

```python
trace_volume.commit()
```

The Modal image dependencies were updated to include:

```python
.uv_pip_install("torch>=2.12.0", "numpy", "matplotlib")
```

This fixed:

- NumPy initialization warning.
- `export_memory_timeline_html failed because matplotlib was not found`.

## First Modal Run With Toy Shapes

The assistant ran:

```bash
uv run modal run iterate.py
```

Initial toy setup:

```text
GPU count: h100:2
world size: 2
B = 2
S = 8
H = 16
num_heads = 4
```

Results:

```text
TP correctness passed.
SP correctness passed.
TSP correctness passed.
```

After adding `numpy` and `matplotlib`, the Modal Volume contained:

```text
tp/tp_attention_rank_0_memory.html
tp/tp_attention_rank_0_trace.json
tp/tp_attention_rank_1_memory.html
tp/tp_attention_rank_1_trace.json

sp/sp_attention_rank_0_memory.html
sp/sp_attention_rank_0_trace.json
sp/sp_attention_rank_1_memory.html
sp/sp_attention_rank_1_trace.json

tsp/tsp_attention_rank_0_memory.html
tsp/tsp_attention_rank_0_trace.json
tsp/tsp_attention_rank_1_memory.html
tsp/tsp_attention_rank_1_trace.json
```

Toy peak allocated memory:

```text
TP:  0.031259 GiB
SP:  0.031264 GiB
TSP: 0.031267 GiB
```

## Update To Paper-Tested Shapes

The user asked to update shapes to match the ones tested in the paper and rerun.

The assistant changed attention benchmark defaults to:

```text
B = 1
S = 8192
H = 4096
num_heads = 32
D = world_size = 8
```

`iterate.py` was updated to:

```python
GPU_CONFIG = "h100:8"
NPROC_PER_NODE = 8
BENCH_B = 1
BENCH_S = 8192
BENCH_H = 4096
BENCH_NUM_HEADS = 32
```

The scripts were updated to accept:

```text
EXPECTED_WORLD_SIZE
BENCH_B
BENCH_S
BENCH_H
BENCH_NUM_HEADS
```

and to validate `world_size == EXPECTED_WORLD_SIZE`.

The user asked if this ensures all GPUs are used. The assistant explained:

- Modal requests `h100:8`.
- `torchrun` launches `--nproc_per_node=8`.
- each rank binds to its `LOCAL_RANK`.
- the scripts validate `EXPECTED_WORLD_SIZE=8`.

## 8-GPU Paper-Shape Modal Run

The user asked to run with the new expected world size.

The assistant ran:

```bash
uv run modal run iterate.py
```

Modal run URL:

```text
https://modal.com/apps/aviralmansingka/main/ap-ENsSwpkEavO22QUQhqUJpn
```

Configuration:

```text
GPU_CONFIG = h100:8
NPROC_PER_NODE = 8
EXPECTED_WORLD_SIZE = 8
B = 1
S = 8192
H = 4096
num_heads = 32
```

Results:

```text
TP correctness passed.
SP correctness passed.
TSP correctness passed.
```

Peak allocated memory:

```text
TP:  0.726562 GiB
SP:  1.132881 GiB
TSP: 0.697334 GiB
```

Volume validation confirmed rank `0..7` files for each mode:

```text
tp/
  tp_attention_rank_{0..7}_trace.json
  tp_attention_rank_{0..7}_memory.html

sp/
  sp_attention_rank_{0..7}_trace.json
  sp_attention_rank_{0..7}_memory.html

tsp/
  tsp_attention_rank_{0..7}_trace.json
  tsp_attention_rank_{0..7}_memory.html
```

The remaining warning:

```text
FutureWarning: export_memory_timeline is deprecated
```

The files were still generated successfully.

## Current Known Caveats

- The paper-scale correctness path still computes a dense full-reference
  attention result for validation, which adds overhead and may distort runtime
  comparisons.
- The memory numbers reported so far are for forward attention calls with
  simplified standalone implementations, not full training.
- The PyTorch memory timeline export API is deprecated, but still works in the
  current run.
- For serious reproduction, separate correctness validation from profiling:
  validate on smaller shapes, then profile paper-scale shapes without dense
  reference computations inside the same job.

## Current Git/Workspace State

Recent `git status --short` showed:

```text
 M README.md
 M iterate.py
 M sequence_parallelism.py
 M tensor_parallelism.py
 M ts_parallelism.py
?? CLAIMS.md
```

Earlier during the conversation, `profiler_utils.py` existed and was used by the
Modal runs. If it is no longer shown as untracked in later status output, check
whether it was already added, moved, or ignored before continuing.

## Suggested Next Steps

- Download or inspect the rank traces from Modal Volume `tp-attn-traces`.
- Add a mode to disable full-reference correctness for paper-scale profiling.
- Add latency summaries alongside peak memory.
- Add CSV/JSON summary output for:
  - mode
  - rank
  - peak allocated
  - peak reserved
  - elapsed time
  - trace paths
- Consider using PyTorch CUDA memory snapshots instead of
  `export_memory_timeline` in future iterations.

## Suggested Skills

- `teach`: use if continuing conceptual understanding of TP/SP/TSP.
- `handoff`: use if preparing another compact transfer document.
- `github:yeet`: use if the local changes should be committed, pushed, and
  opened as a PR.
