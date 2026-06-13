---
mode: solo
student:
source: https://arxiv.org/pdf/2604.26294; TP+SP versus TSP tradeoffs
started: 2026-06-12T19:46:43-0400
---

# Teaching: TP+SP vs TSP Tradeoffs

## Progress: 14/14 concepts confirmed

### The Problem
- [x] TP+SP uses two separate mesh axes: tensor parallel degree T shards weights/features, sequence parallel degree Sigma shards tokens.
- [x] With fixed total degree D = T * Sigma, TP+SP must split the parallelism budget between model-state memory reduction and activation-memory reduction.
- [x] In TP+SP, weight-proportional memory scales with 1/T, while activation memory scales with 1/Sigma.
- [x] TP+SP inherits both communication families: TP-style activation all-reduces and SP-style K/V exchange.
- [x] TSP is not strictly better in all situations because it adds runtime weight movement and depends on topology and overlap.

### The Solution
- [x] TSP folds TP and SP into one axis of size D, so both weight-proportional memory and activation memory scale by roughly 1/D.
- [x] TSP can avoid needing D^2 devices for simultaneous D-way weight sharding and D-way sequence sharding.
- [x] TSP attention still pays sequence K/V exchange and also moves attention weight shards through broadcast-style steps.
- [x] TSP MLP moves weight shards through a ring while accumulating local-token outputs, trading weight traffic for avoiding TP activation all-reduce.
- [x] TP+SP is attractive when independent control of T and Sigma is valuable, when TP all-reduces are cheap, or when weight movement cannot be hidden.
- [x] TSP is attractive when memory is the bottleneck, context or batch is large, and extra weight communication can stay on fast links or overlap with GEMMs.

### Broader Context
- [x] At short context or small batch, weight/state memory may dominate, so high TP degree or TSP can look similar.
- [x] At long context, activation memory dominates, so SP helps, but TSP can combine TP-like model-state savings with SP-like activation savings.
- [x] The practical choice is hardware-aware: raw communication volume is less important than critical-path communication, topology, bandwidth, latency, and memory headroom.

---
*Last updated: 2026-06-12T20:03:14-0400*
