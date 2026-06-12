---
mode: solo
student:
source: row-parallel vs. column-parallel and how it is relevant to attention; focused on matrix orientation
started: 2026-06-12T19:02:48-0400
---

# Teaching: Row-Parallel vs Column-Parallel Attention Orientation

## Progress: 12/12 concepts confirmed

### The Problem
- [x] Matrix orientation must be fixed first: activations are row vectors and linear layers are written as X @ W.
- [x] With X shaped [B, S, H] and W shaped [H, O], rows of W correspond to input features and columns of W correspond to output features.
- [x] The words row-parallel and column-parallel refer to splitting rows or columns of W, not splitting rows or columns of the activation tensor.
- [x] Confusing weight-axis sharding with activation-axis sharding causes wrong expectations for local output shapes.

### The Solution
- [x] Column-parallel W splits output columns: W_i is [H, O/TP], so replicated X @ W_i gives local output [B, S, O/TP].
- [x] Column-parallel Q/K/V is natural for attention because output columns align with head or head-feature shards.
- [x] Local Q_i, K_i, and V_i for a head shard can compute attention independently when sequence activations are replicated.
- [x] Row-parallel W splits input rows: X_i is [B, S, H/TP] and W_i is [H/TP, O], so X_i @ W_i gives a partial [B, S, O].
- [x] Row-parallel outputs must be summed across ranks with all-reduce because each rank produced a partial contribution to the same output coordinates.
- [x] Attention uses column-parallel Q/K/V followed by row-parallel Wo so local heads are computed first, then their residual-space contributions are summed.

### Broader Context
- [x] Column-parallel produces sharded activations; row-parallel consumes sharded activations and produces summed full-width activations.
- [x] Matrix orientation conventions can flip the visual intuition, so always derive from the actual equation and tensor shapes.

---
*Last updated: 2026-06-12T20:01:00-0400*
