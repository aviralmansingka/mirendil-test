---
mode: solo
student:
source: https://arxiv.org/pdf/2604.26294 Algorithm 1; TSP attention implementation
started: 2026-06-12T20:05:27-0400
---

# Teaching: TSP Attention Implementation

## Progress: 18/18 concepts confirmed

### The Problem
- [x] TSP attention starts with each rank owning a local sequence shard X shaped [B, S/D, H] and only one shard of the attention weights.
- [x] The paper writes local Wq/Wk/Wv shards as [H/D, H] and multiplies by W^T, which is equivalent to column-parallel output-feature sharding under X @ W.
- [x] A local rank cannot compute full local-token attention from only its own weight shard because that covers only one head or feature shard.
- [x] A local rank cannot compute correct causal attention from only its local K/V sequence shard because queries need K/V from other sequence shards.
- [x] Plain TP would use all-reduce after row-parallel Wo, but TSP changes ownership: each rank owns final outputs only for its local tokens.

### The Solution
- [x] TSP loops over weight-owning ranks r so every sequence shard applies every attention weight shard to its local tokens.
- [x] Broadcast sends the packed Wq/Wk/Wv/Wo shard from owner r to all ranks for the current iteration.
- [x] After broadcast, each rank computes local Q/K/V for the current head shard from its local X.
- [x] For the current head bucket, all-gather collects K/V across the sequence axis so local Q can attend over the global sequence context.
- [x] Zigzag reorder restores gathered K/V into logical sequence order before applying causal attention.
- [x] Causal bounds or masks ensure each local query token only attends to allowed prefix tokens.
- [x] The local attention output A for the current head shard has shape [B, S/D, H/D] before Wo.
- [x] The current Wo shard is row-parallel with respect to head-space input features, so A @ Wo_r contributes a partial [B, S/D, H] output for local tokens.
- [x] TSP accumulates those [B, S/D, H] contributions locally across all r instead of all-reducing them across ranks.
- [x] All-reduce is not the right final primitive in TSP attention because outputs are sequence-sharded, not replicated full-sequence tensors.

### Broader Context
- [x] Broadcast moves weight shards; all-gather moves activation K/V shards; all-reduce is avoided in the forward attention output path.
- [x] Head bucketing trades peak K/V memory for repeated all-gathers and makes the algorithm implementable at long context.
- [x] Deriving the algorithm from ownership rules prevents confusion: ask who owns weights, who owns tokens, and whether tensors are partial sums or distinct shards.

---
*Last updated: 2026-06-12T20:47:08-0400*
