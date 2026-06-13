from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function


@contextmanager
def profile_rank_region(
    label: str,
    device: torch.device,
    trace_dir: str | None = None,
) -> Iterator[None]:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    local_rank = device.index if device.index is not None else torch.cuda.current_device()
    trace_dir = trace_dir or os.environ.get("TRACE_DIR", "traces")

    os.makedirs(trace_dir, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function(f"{label}_rank_{rank}"):
            start_event.record()
            yield
            end_event.record()
        torch.cuda.synchronize(device)

    elapsed_ms = start_event.elapsed_time(end_event)
    peak_allocated_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3

    trace_path = os.path.join(trace_dir, f"{label}_rank_{rank}_trace.json")
    memory_path = os.path.join(trace_dir, f"{label}_rank_{rank}_memory.html")
    prof.export_chrome_trace(trace_path)
    prof.export_memory_timeline(memory_path, device=f"cuda:{local_rank}")

    peak_allocated = torch.tensor(peak_allocated_gb, device=device)
    max_elapsed_ms = torch.tensor(elapsed_ms, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)
        dist.all_reduce(max_elapsed_ms, op=dist.ReduceOp.MAX)

    print(
        f"[rank {rank}] {label}: "
        f"elapsed_ms={elapsed_ms:.3f}, "
        f"max_rank_elapsed_ms={max_elapsed_ms.item():.3f}, "
        f"peak_allocated={peak_allocated_gb:.6f} GiB, "
        f"peak_reserved={peak_reserved_gb:.6f} GiB, "
        f"max_rank_peak_allocated={peak_allocated.item():.6f} GiB, "
        f"trace={trace_path}, memory={memory_path}",
        flush=True,
    )
