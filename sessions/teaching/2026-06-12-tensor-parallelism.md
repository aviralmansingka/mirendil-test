---
mode: solo
student:
source: tensor parallelism, with context from arXiv:2604.26294
started: 2026-06-12T18:26:03-0400
---

# Teaching: Tensor Parallelism for Attention

## Progress: 0/12 concepts confirmed

### The Problem
- [ ] Attention uses large learned projection matrices Wq, Wk, Wv, and Wo.
- [ ] Replicating all attention weights on every GPU wastes memory when the model is large.
- [ ] In plain TP attention, the input sequence activations are usually replicated across the TP group.

### The Solution
- [ ] Column-parallel Q/K/V splits output features or heads across ranks, so each rank computes a subset of Q, K, and V.
- [ ] Head-wise attention lets each rank compute attention independently for its local heads after column-parallel Q/K/V.
- [ ] Row-parallel Wo splits input features or heads across ranks, so each rank computes a partial contribution to the residual dimension.
- [ ] An all-reduce sums row-parallel Wo partial outputs so every rank receives the full attention output.
- [ ] An all-gather concatenates sharded tensor pieces when every rank needs the full tensor rather than a summed result.
- [ ] A broadcast copies one rank's tensor to all other ranks, useful when one owner has data that every peer must consume.

### Broader Context
- [ ] TP attention reduces per-rank attention weight memory and per-rank projection compute roughly with TP degree.
- [ ] TP attention communication is activation-sized and grows with batch size, sequence length, and hidden size.
- [ ] TSP changes the standard TP attention story by also sharding sequence tokens and exchanging K/V context.

---
*Last updated: 2026-06-12T18:31:00-0400*
