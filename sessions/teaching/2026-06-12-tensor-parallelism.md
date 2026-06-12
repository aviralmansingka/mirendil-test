---
mode: solo
student:
source: tensor parallelism, with context from arXiv:2604.26294
started: 2026-06-12T18:26:03-0400
---

# Teaching: Tensor Parallelism

## Progress: 0/10 concepts confirmed

### The Problem
- [ ] A single GPU may not have enough memory or compute bandwidth for a large transformer layer.
- [ ] Transformer linear layers are dominated by large matrix multiplications over model weights.
- [ ] Long-context training can shift the bottleneck from model weights to activations.

### The Solution
- [ ] Tensor parallelism shards weight matrices across multiple GPUs.
- [ ] Each GPU computes a partial result from its local weight shard.
- [ ] GPUs communicate to reconcile partial results into the correct full layer output.
- [ ] Column-parallel and row-parallel linear layers split different matrix dimensions and require different communication.

### Broader Context
- [ ] TP reduces parameter, gradient, optimizer-state, and some per-device compute cost roughly with TP degree.
- [ ] TP keeps sequence activations replicated, which matters for long context.
- [ ] TSP in the paper folds TP with sequence parallelism so the same GPU group shards both weights and tokens.

---
*Last updated: 2026-06-12T18:26:03-0400*
