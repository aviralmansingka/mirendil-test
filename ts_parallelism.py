import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("this script requires CUDA")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "launch with: torchrun --standalone --nproc_per_node=2 tp.py"
        )

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"expected world_size == 2, got {world_size}")

    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    if LOCAL_RANK >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={LOCAL_RANK} but only {torch.cuda.device_count()} CUDA devices exist"
        )

    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda", LOCAL_RANK)
    return rank, world_size, LOCAL_RANK, device


@contextmanager
def distributed_context() -> Iterator[tuple[int, int, int, torch.device]]:
    try:
        yield setup_distributed()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main():
    with distributed_context() as (rank, world_size, _LOCAL_RANK, device):
        pass


if __name__ == "__main__":
    main()
