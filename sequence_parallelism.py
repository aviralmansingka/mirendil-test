from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F


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


def project_to_heads(
    projected: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    batch_size, seq_len, _hidden_size = projected.shape
    return projected.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def init_tensor(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    fan_in = shape[-1]
    scale = fan_in**-0.5
    return torch.randn(*shape, device=device) * scale


def assert_sp_config(seq_len: int, world_size: int) -> None:
    if seq_len % world_size != 0:
        raise ValueError("seq_len must be divisible by world_size")


def assert_mlp_config(intermediate_size: int, world_size: int) -> None:
    if intermediate_size % world_size != 0:
        raise ValueError("intermediate_size must be divisible by world_size")


def shard_seq_parallel(X: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    seq_len = X.shape[-2]
    assert_sp_config(seq_len, world_size)

    local_seq = seq_len // world_size
    start = rank * local_seq
    end = (rank + 1) * local_seq
    return X[:, start:end, :].contiguous()


def shard_row_parallel(
    W_o: torch.Tensor,
    rank: int,
    world_size: int,
    num_heads: int,
) -> torch.Tensor:
    H_out, H_in = W_o.shape
    if H_in % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    if num_heads % world_size != 0:
        raise ValueError("num_heads must be divisible by world_size")
    if H_out != H_in:
        raise ValueError(f"expected square output weight, got {tuple(W_o.shape)}")

    D = H_in // num_heads
    p_heads = num_heads // world_size
    start = rank * p_heads * D
    end = start + p_heads * D
    return W_o[:, start:end].contiguous()


def shard_mlp_col_parallel(
    W: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    I, _H = W.shape
    assert_mlp_config(I, world_size)
    p_I = I // world_size
    start = rank * p_I
    end = start + p_I
    return W[start:end, :].contiguous()


def shard_mlp_row_parallel(
    W: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    _H, I = W.shape
    assert_mlp_config(I, world_size)
    p_I = I // world_size
    start = rank * p_I
    end = start + p_I
    return W[:, start:end].contiguous()


def broadcast_from_rank0(tensor: torch.Tensor) -> torch.Tensor:
    dist.broadcast(tensor, src=0)
    return tensor


def attn(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    B, S, H = X.shape
    if H % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    D = H // num_heads

    # q,k,v: [B, num_heads, S, D]
    q = project_to_heads(F.linear(X, W_q), num_heads, D)
    k = project_to_heads(F.linear(X, W_k), num_heads, D)
    v = project_to_heads(F.linear(X, W_v), num_heads, D)

    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
    )
    # attn: [B, num_heads, S, D]
    attn = attn.transpose(1, 2).contiguous().view(B, S, H)
    return F.linear(attn, W_o)


def mlp(
    X: torch.Tensor,
    W_in: torch.Tensor,
    W_out: torch.Tensor,
) -> torch.Tensor:
    hidden = F.gelu(F.linear(X, W_in))
    return F.linear(hidden, W_out)


def assert_shapes(
    H: int,
    W_q_p: torch.Tensor,
    W_k_p: torch.Tensor,
    W_v_p: torch.Tensor,
    W_o_p: torch.Tensor,
    p_heads: int,
) -> int:
    p_hidden = W_q_p.shape[0]
    if W_k_p.shape != W_q_p.shape or W_v_p.shape != W_q_p.shape:
        raise ValueError("p Q, K, and V weights must have matching shapes")
    if W_o_p.shape != (H, p_hidden):
        raise ValueError(
            f"expected W_o_p shape {(H, p_hidden)}, got {tuple(W_o_p.shape)}"
        )
    if p_hidden % p_heads != 0:
        raise ValueError("p_hidden must be divisible by p_heads")
    return p_hidden


def assert_mlp_shapes(
    H: int,
    W_in_p: torch.Tensor,
    W_out_p: torch.Tensor,
) -> int:
    p_I, H_in = W_in_p.shape
    if H_in != H:
        raise ValueError(f"expected W_in_p input size {H}, got {H_in}")
    if W_out_p.shape != (H, p_I):
        raise ValueError(
            f"expected W_out_p shape {(H, p_I)}, got {tuple(W_out_p.shape)}"
        )
    return p_I


def sp_attn(
    X_p: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    B, local_seq, H = X_p.shape
    if H % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    D = H // num_heads

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_seq = local_seq * world_size

    # q/k/v: [B, num_heads, local_S, D]
    q = project_to_heads(F.linear(X_p, W_q), num_heads, D)
    k_p = project_to_heads(F.linear(X_p, W_k), num_heads, D).contiguous()
    v_p = project_to_heads(F.linear(X_p, W_v), num_heads, D).contiguous()

    gathered_k = [torch.empty_like(k_p) for _ in range(world_size)]
    gathered_v = [torch.empty_like(v_p) for _ in range(world_size)]
    dist.all_gather(gathered_k, k_p)
    dist.all_gather(gathered_v, v_p)

    # k/v: [B, num_heads, S, D]
    k = torch.cat(gathered_k, dim=2)
    v = torch.cat(gathered_v, dim=2)

    q_positions = torch.arange(
        rank * local_seq,
        (rank + 1) * local_seq,
        device=X_p.device,
    )
    k_positions = torch.arange(total_seq, device=X_p.device)
    causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)

    attn_p = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=causal_mask,
        dropout_p=0.0,
    )
    # attn_p: [B, num_heads, local_S, D]
    attn_p = attn_p.transpose(1, 2).contiguous().view(B, local_seq, H)

    return F.linear(attn_p, W_o)


def sp_mlp(
    X_p: torch.Tensor,
    W_in: torch.Tensor,
    W_out: torch.Tensor,
) -> torch.Tensor:
    _B, _S, H = X_p.shape
    assert_mlp_shapes(H, W_in, W_out)

    hidden_p = F.gelu(F.linear(X_p, W_in))
    out = F.linear(hidden_p, W_out)
    return out


def manual_attn(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    B, S, H = X.shape
    if H % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    D = H // num_heads

    q = project_to_heads(F.linear(X, W_q), num_heads, D)
    k = project_to_heads(F.linear(X, W_k), num_heads, D)
    v = project_to_heads(F.linear(X, W_v), num_heads, D)

    # q: [B, num_heads, S, D]
    # k: [B, num_heads, S, D]
    # scores: [B, num_heads, S, S]
    scale = D**-0.5
    scores = q @ k.transpose(-2, -1) * scale
    causal_mask = torch.ones(S, S, device=X.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal_mask, float("-inf"))

    # v: [B, num_heads, S, D]
    # attn: [B, num_heads, S, D]
    attn = torch.softmax(scores, dim=-1) @ v
    attn = attn.transpose(1, 2).contiguous().view(B, S, H)
    return F.linear(attn, W_o)


def run_sp_attn(
    device: torch.device, rank: int, world_size: int, should_print: bool
) -> None:
    B = 2
    S = 8
    H = 16
    num_heads = 4
    if H % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    assert_sp_config(S, world_size)

    if rank == 0:
        torch.manual_seed(1234)
        W_q = init_tensor((H, H), device)
        W_k = init_tensor((H, H), device)
        W_v = init_tensor((H, H), device)
        W_o = init_tensor((H, H), device)

        torch.manual_seed(5678)
        X = torch.randn(B, S, H, device=device)
    else:
        W_q = torch.empty(H, H, device=device)
        W_k = torch.empty(H, H, device=device)
        W_v = torch.empty(H, H, device=device)
        W_o = torch.empty(H, H, device=device)
        X = torch.empty(B, S, H, device=device)

    # W_q/W_k/W_v/W_o: [H, H]
    W_q = broadcast_from_rank0(W_q)
    W_k = broadcast_from_rank0(W_k)
    W_v = broadcast_from_rank0(W_v)
    W_o = broadcast_from_rank0(W_o)
    # [B, S, H]
    X = broadcast_from_rank0(X)

    # [B, S/D, H]
    X_p = shard_seq_parallel(X, rank, world_size)

    with torch.no_grad():
        expected = attn(X, W_q, W_k, W_v, W_o, num_heads)
        out = sp_attn(
            X_p,
            W_q,
            W_k,
            W_v,
            W_o,
            num_heads,
        )

    expected_p = shard_seq_parallel(expected, rank, world_size)
    torch.testing.assert_close(out, expected_p, rtol=1e-5, atol=1e-5)

    gathered = [torch.empty_like(out) for _ in range(world_size)]
    dist.all_gather(gathered, out)
    gathered_out = torch.cat(gathered, dim=1)
    torch.testing.assert_close(gathered_out, expected, rtol=1e-5, atol=1e-5)

    if should_print:
        print("sequence-parallel causal attention correctness check passed.")


def run_sp_mlp(
    device: torch.device, rank: int, world_size: int, should_print: bool
) -> torch.Tensor:
    B = 2
    S = 8
    H = 16
    I = 64

    if rank == 0:
        torch.manual_seed(1234)
        W_in = init_tensor((I, H), device)
        W_out = init_tensor((H, I), device)

        torch.manual_seed(5678)
        X = torch.randn(B, S, H, device=device)
    else:
        W_in = torch.empty(I, H, device=device)
        W_out = torch.empty(H, I, device=device)
        X = torch.empty(B, S, H, device=device)

    # W_in: [I, H]
    # W_out: [H, I]
    W_in = broadcast_from_rank0(W_in)
    W_out = broadcast_from_rank0(W_out)
    # X: [B, S, H]
    X = broadcast_from_rank0(X)

    X_p = shard_seq_parallel(X, rank, world_size)

    with torch.no_grad():
        expected = mlp(X, W_in, W_out)
        out_p = sp_mlp(X_p, W_in, W_out)

    gathered = [torch.empty_like(out_p) for _ in range(world_size)]
    dist.all_gather(gathered, out_p)

    out = torch.cat(gathered, dim=1)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    if should_print:
        print("sequence-parallel MLP correctness check passed.")

    return out


def main() -> None:
    with distributed_context() as (rank, world_size, _LOCAL_RANK, device):
        run_sp_attn(device, rank, world_size, should_print=rank == 0)
        run_sp_mlp(device, rank, world_size, should_print=rank == 0)


if __name__ == "__main__":
    main()
