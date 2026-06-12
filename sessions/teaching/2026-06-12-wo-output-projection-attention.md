---
mode: solo
student:
source: Wo output projection in attention, based on the tensor-parallel attention teaching session
started: 2026-06-12T18:49:49-0400
---

# Teaching: Wo Output Projection in Attention

## Progress: 9/9 concepts confirmed

### The Problem
- [x] Multi-head attention produces per-head or concatenated head outputs before the output projection.
- [x] The output projection Wo maps concatenated attention features back into the residual hidden space.
- [x] Confusing head-sharded tensors with residual-space tensors leads to wrong shape expectations.

### The Solution
- [x] In full attention, A has shape [B, S, H], Wo has shape [H, H], and A @ Wo has shape [B, S, H].
- [x] In 4-way TP, local attention output A_i has shape [B, S, H/4].
- [x] Row-parallel Wo_i has shape [H/4, H], so A_i @ Wo_i has shape [B, S, H].
- [x] Each rank's [B, S, H] after Wo_i is only a partial contribution, not the final output.
- [x] An all-reduce sum combines partial [B, S, H] contributions into the final attention output on each rank.

### Broader Context
- [x] The output projection mixes information from all local heads into every residual hidden dimension.

---
*Last updated: 2026-06-12T19:24:00-0400*
