---
mode: solo
student:
source: https://arxiv.org/pdf/2604.26294; sequence parallelism and TSP tradeoffs
started: 2026-06-12T19:31:08-0400
---

# Teaching: Sequence Parallelism and TSP Tradeoffs

## Progress: 0/13 concepts confirmed

### The Problem
- [ ] In plain tensor parallelism, inputs and sequence activations remain replicated across the TP group.
- [ ] Sequence parallelism shards the activation tensor along the sequence/token axis, so each rank owns only a subset of tokens.
- [ ] Sharding sequence reduces activation memory and attention work per rank, which matters more as context length grows.
- [ ] Sequence parallelism keeps model weights replicated, so it does not reduce parameter, gradient, or optimizer-state memory.
- [ ] Attention is the hard sequence-parallel operation because each query token may need keys and values from tokens stored on other ranks.

### The Solution
- [ ] Sequence-parallel attention can recover global context by exchanging K/V shards using a ring, all-gather, or all-to-all style schedule.
- [ ] A naive contiguous sequence split causes causal-attention load imbalance because later tokens attend to longer prefixes.
- [ ] Zigzag or striped sequence partitioning balances causal attention by giving each rank a mix of early and late tokens.
- [ ] TSP folds tensor parallelism and sequence parallelism onto one device axis, so each rank owns both a weight shard and a sequence shard.
- [ ] In TSP attention, ranks iterate over broadcast attention weight shards and all-gather K/V for the current shard to compute correct local-token attention.
- [ ] In TSP MLP, weight shards can rotate through ranks while each rank accumulates outputs for its local tokens, avoiding a standard TP activation all-reduce.

### Broader Context
- [ ] TSP reduces both model-state memory and activation memory by roughly the folded degree, but adds runtime weight movement.
- [ ] The paper argues TSP is most attractive when memory is the bottleneck, context is long, and extra communication can stay on fast links or overlap with compute.

---
*Last updated: 2026-06-12T19:31:08-0400*
