# Tensor Sequence Parallelism


## Motivation

Transformer workloads are limited not only by parameter memory, but also by the activation memory that grows with
sequence length. Especially as new releases increasingly focus on longer context serving (e.g. SubQ with 12M context),
it's increasingly important to support efficient training for these scenarios.

While existing tensor parallel approaches reduce the per-device weight and compute burden, they still require each
device to hold the full sequence activations for many operations. Sequence parallelism addresses that activation
pressure by sharding tokens across devices, but it comes with its own tradeoff for keeping all model weights on device.

The paper combines both dimensions so memory is reduced along the model and sequence axes together. This report
validates whether that combined strategy preserves correctness while lowering peak per-device memory in attention and
MLP workloads.

## Implementation

The implementation evaluates three baselines: tensor parallelism (TP), sequence parallelism (SP), and a combined TSP
approach. Each implementation covers causal attention and the MLP sublayer. The attention implementation is particularly
important because sequence partitioning requires each rank to recover global key/value context while still applying the
causal mask for its local query positions.

## Validation

- Build a single-device reference output.
- Run the parallel implementation with the same inputs and weights.
- Compare each local output against the corresponding reference shard.
- Gather distributed outputs and compare the reconstructed output against the full reference output.

## Experimental Setup

- GPU count: 8
- GPU type: H100
- Batch size: 8
- Sequence length: 8K, 16K, 32K, 64K
- Hidden size: 2K
- Number of attention heads: 16, 32

## Results

### Correctness

The completed TP, SP, and TSP runs in `log.txt` pass the causal attention and MLP correctness checks against the
single-device reference implementation with a standard atol/rtol of 1e-5.

### Peak Memory

The following values are the maximum per-rank peak allocated memory reported by the profiler for the attention region.

| Sequence Length | Hidden Size | Heads | Mode | Peak Allocated | Peak Reserved |
|---:|---:|---:|---|---:|---:|
| 8K | 2K | 16 | TP | 1.845703 GiB | 3.115234 GiB |
| 8K | 2K | 16 | SP | 2.843750 GiB | 3.173828 GiB |
| 8K | 2K | 16 | TSP | 1.580147 GiB | 3.177734 GiB |
| 8K | 2K | 32 | TP | 1.845703 GiB | 3.115234 GiB |
| 8K | 2K | 32 | SP | 2.843750 GiB | 3.173828 GiB |
| 8K | 2K | 32 | TSP | 1.580147 GiB | 3.177734 GiB |
| 16K | 2K | 16 | TP | 3.595703 GiB | 6.115234 GiB |
| 16K | 2K | 16 | SP | 5.593750 GiB | 6.236328 GiB |
| 16K | 2K | 16 | TSP | 3.072403 GiB | 6.240234 GiB |
| 16K | 2K | 32 | TP | 3.595703 GiB | 6.115234 GiB |
| 16K | 2K | 32 | SP | 5.593750 GiB | 6.236328 GiB |
| 16K | 2K | 32 | TSP | 3.072403 GiB | 6.240234 GiB |
| 32K | 2K | 16 | TP | 7.095703 GiB | 12.115234 GiB |
| 32K | 2K | 16 | SP | 11.093750 GiB | 12.361328 GiB |
| 32K | 2K | 16 | TSP | 6.135040 GiB | 12.365234 GiB |
| 32K | 2K | 32 | TP | 7.095703 GiB | 12.115234 GiB |
| 32K | 2K | 32 | SP | 11.093750 GiB | 12.361328 GiB |
| 32K | 2K | 32 | TSP | 6.135040 GiB | 12.365234 GiB |
| 64K | 2K | 16 | TP | 14.095703 GiB | 24.115234 GiB |
| 64K | 2K | 16 | SP | 22.093750 GiB | 24.611328 GiB |
| 64K | 2K | 16 | TSP | 13.416565 GiB | 24.615234 GiB |
| 64K | 2K | 32 | TP | 14.095703 GiB | 24.115234 GiB |
| 64K | 2K | 32 | SP | 22.093750 GiB | 24.611328 GiB |
| 64K | 2K | 32 | TSP | 13.416565 GiB | 24.615234 GiB |

### Runtime

The reported runtime uses `max_rank_elapsed_ms` as the distributed step latency for the profiled attention region. Mean
rank elapsed time is included to show rank imbalance, but the max-rank value is the relevant end-to-end timing.

| Sequence Length | Hidden Size | Heads | Mode | Max Rank Time | Mean Rank Time |
|---:|---:|---:|---|---:|---:|
| 8K | 2K | 16 | TP | 19.339 ms | 16.362 ms |
| 8K | 2K | 16 | SP | 61.433 ms | 34.208 ms |
| 8K | 2K | 16 | TSP | 35.252 ms | 32.375 ms |
| 8K | 2K | 32 | TP | 53.433 ms | 30.405 ms |
| 8K | 2K | 32 | SP | 80.332 ms | 47.695 ms |
| 8K | 2K | 32 | TSP | 219.216 ms | 72.331 ms |
| 16K | 2K | 16 | TP | 52.819 ms | 44.008 ms |
| 16K | 2K | 16 | SP | 149.002 ms | 96.628 ms |
| 16K | 2K | 16 | TSP | 123.140 ms | 97.724 ms |
| 16K | 2K | 32 | TP | 64.832 ms | 54.893 ms |
| 16K | 2K | 32 | SP | 192.858 ms | 129.474 ms |
| 16K | 2K | 32 | TSP | 178.973 ms | 126.682 ms |
| 32K | 2K | 16 | TP | 138.952 ms | 133.162 ms |
| 32K | 2K | 16 | SP | 450.145 ms | 296.286 ms |
| 32K | 2K | 16 | TSP | 285.154 ms | 280.647 ms |
| 32K | 2K | 32 | TP | 168.837 ms | 164.969 ms |
| 32K | 2K | 32 | SP | 489.568 ms | 450.039 ms |
| 32K | 2K | 32 | TSP | 438.466 ms | 429.372 ms |
| 64K | 2K | 16 | TP | 536.169 ms | 467.549 ms |
| 64K | 2K | 16 | SP | 988.241 ms | 965.202 ms |
| 64K | 2K | 16 | TSP | 1022.820 ms | 1014.361 ms |
| 64K | 2K | 32 | TP | 583.438 ms | 572.961 ms |
| 64K | 2K | 32 | SP | 1732.589 ms | 1701.377 ms |
| 64K | 2K | 32 | TSP | 1892.692 ms | 1722.828 ms |

### TSP MLP Runtime And Memory

The latest run also profiles the TSP MLP region. TP and SP MLP baselines are currently correctness-checked but not
profiled as separate regions in `log.txt`.

| Sequence Length | Hidden Size | Heads | Mode | Max Rank Time | Mean Rank Time | Peak Allocated | Peak Reserved |
|---:|---:|---:|---|---:|---:|---:|---:|
| 8K | 2K | 16 | TSP | 14.933 ms | 14.105 ms | 1.460938 GiB | 8.427734 GiB |
| 8K | 2K | 32 | TSP | 14.153 ms | 13.662 ms | 1.460938 GiB | 8.427734 GiB |
| 16K | 2K | 16 | TSP | 26.426 ms | 25.712 ms | 2.742188 GiB | 16.740234 GiB |
| 16K | 2K | 32 | TSP | 26.572 ms | 25.491 ms | 2.742188 GiB | 16.740234 GiB |
| 32K | 2K | 16 | TSP | 49.415 ms | 48.636 ms | 5.304688 GiB | 33.365234 GiB |
| 32K | 2K | 32 | TSP | 48.747 ms | 48.313 ms | 5.304688 GiB | 33.365234 GiB |
| 64K | 2K | 16 | TSP | 111.469 ms | 110.599 ms | 10.429688 GiB | 66.615234 GiB |
| 64K | 2K | 32 | TSP | 113.271 ms | 112.564 ms | 10.429688 GiB | 66.615234 GiB |


## Limitations

- Sequence Parallelism was implemented with all-gather for KV
  - This greatly increased peak memory usage
  - Future work would include leveraging ring-attention to reduce memory usage
- TSP was implemented without async overlap
  - This impacted performance of the TSP approach

## Findings

TODO: Convert the raw results into conclusions. Suggested findings to evaluate:

- Whether TSP reduces peak memory relative to TP and SP.
- Whether the reduction is large enough to justify the extra communication.
- Which operation is the bottleneck: attention, MLP, or collectives.
- Whether the implementation behaves as predicted by the paper.

## AI Usage

Codex was used extensively throughout this task. It was first used via the $teach skill to understand the paper over the
course of various sessions (available in this repo). The $grill-me skill was also used for bootstrapping initial
baselines.

In terms of implementation, Codex was used for establishing the TP and SP baselines as well as steering implementation
choices to keep the code readable to iterate on. In particular, it was used to also generate profiling and distributed
code scaffolding used through the repo

The actual TSP implementation for attn and MLP was written __artisanally__. The purpose was to test my own understanding
through the implementation to be able to analyze the results effectively.

All conversations regarding this project are also included in the `codex/` folder

## Future Work

- Avoid repeated full-weight broadcasts by keeping sharded parameters resident on each rank.
- Overlap communication with local projection, attention, and MLP compute.
- Replace blocking collectives with asynchronous or fused communication where appropriate.
- Evaluate FlashAttention-compatible sequence partitioning.
- Add activation checkpointing trials.
- Sweep tensor-parallel and sequence-parallel degrees.
- Test multi-node topology-aware group placement.
- Extend the benchmark from isolated layers to full Transformer blocks and training steps.
