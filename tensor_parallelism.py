from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

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

    dist.init_process_group(backend="nccl")

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


def assert_tp_config(H: int, num_heads: int, world_size: int) -> None:
    if H % num_heads != 0:
        raise ValueError("H must be divisible by num_heads")
    if num_heads % world_size != 0:
        raise ValueError("num_heads must be divisible by world_size")


def assert_mlp_tp_config(intermediate_size: int, world_size: int) -> None:
    if intermediate_size % world_size != 0:
        raise ValueError("intermediate_size must be divisible by world_size")


def shard_col_parallel(
    W: torch.Tensor,
    rank: int,
    world_size: int,
    num_heads: int,
) -> torch.Tensor:
    H_out, H_in = W.shape
    assert_tp_config(H_out, num_heads, world_size)
    if H_out != H_in:
        raise ValueError(f"expected square QKV weight, got {tuple(W.shape)}")

    D = H_out // num_heads
    p_heads = num_heads // world_size
    start = rank * p_heads * D
    end = start + p_heads * D
    return W[start:end, :].contiguous()


def shard_row_parallel(
    W_o: torch.Tensor,
    rank: int,
    world_size: int,
    num_heads: int,
) -> torch.Tensor:
    H_out, H_in = W_o.shape
    assert_tp_config(H_in, num_heads, world_size)
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
    assert_mlp_tp_config(I, world_size)
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
    assert_mlp_tp_config(I, world_size)
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
    if W_k_p.shape != W_q_p.shape or W_v_local.shape != W_q_local.shape:
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
            f"expected W_out_p shape {(H, p_I)}, got {tuple(W_out_local.shape)}"
        )
    return p_I


def tp_attn(
    X: torch.Tensor,
    W_q_p: torch.Tensor,
    W_k_p: torch.Tensor,
    W_v_p: torch.Tensor,
    W_o_p: torch.Tensor,
    p_heads: int,
) -> torch.Tensor:
    B, S, H = X.shape
    p_hidden = assert_shapes(
        H,
        W_q_p,
        W_k_p,
        W_v_p,
        W_o_p,
        p_heads,
    )
    D = p_hidden // p_heads

    # q,k,v: [B, p_heads, S, D]
    q = project_to_heads(F.linear(X, W_q_p), p_heads, D)
    k = project_to_heads(F.linear(X, W_k_p), p_heads, D)
    v = project_to_heads(F.linear(X, W_v_p), p_heads, D)

    attn_p = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
    )
    # attn_p: [B, p_heads, S, D]
    attn_p = attn_p.transpose(1, 2).contiguous().view(B, S, local_hidden)

    out = F.linear(attn_p, W_o_p)
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def tp_mlp(
    X: torch.Tensor,
    W_in_p: torch.Tensor,
    W_out_p: torch.Tensor,
) -> torch.Tensor:
    _B, _S, H = X.shape
    assert_mlp_shapes(H, W_in_p, W_out_p)

    hidden_p = F.gelu(F.linear(X, W_in_p))
    out = F.linear(hidden_p, W_out_p)
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
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


def run_tp_attn(
    device: torch.device, rank: int, world_size: int, should_print: bool
) -> None:
    B = 2
    S = 8
    H = 16
    num_heads = 4
    assert_tp_config(H, num_heads, world_size)

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

    # broadcast entire weights to all gpus
    # W_q/W_k/W_v/W_o: [H, H]
    W_q = broadcast_from_rank0(W_q)
    W_k = broadcast_from_rank0(W_k)
    W_v = broadcast_from_rank0(W_v)
    W_o = broadcast_from_rank0(W_o)
    # [B, S, H]
    X = broadcast_from_rank0(X)

    p_heads = num_heads // world_size
    # W_q_p/W_k_p/W_v_local: [local_hidden, H]
    W_q_p = shard_col_parallel(W_q, rank, world_size, num_heads)
    W_k_p = shard_col_parallel(W_k, rank, world_size, num_heads)
    W_v_p = shard_col_parallel(W_v, rank, world_size, num_heads)
    # W_o_p: [H, p_hidden]
    W_o_p = shard_row_parallel(W_o, rank, world_size, num_heads)

    with torch.no_grad():
        expected = attn(X, W_q, W_k, W_v, W_o, num_heads)
        out = tp_attn(
            X,
            W_q_p,
            W_k_p,
            W_v_p,
            W_o_p,
            p_heads,
        )

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    gathered = [torch.empty_like(out) for _ in range(world_size)]
    dist.all_gather(gathered, out)
    for other_rank, other_out in enumerate(gathered):
        torch.testing.assert_close(other_out, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(other_out, out, rtol=1e-5, atol=1e-5)

    if should_print:
        print("tensor-parallel causal attention correctness check passed.")


def run_tp_mlp(
    device: torch.device, rank: int, world_size: int, should_print: bool
) -> torch.Tensor:
    B = 2
    S = 8
    H = 16
    I = 64
    assert_mlp_tp_config(I, world_size)

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

    # W_in_p: [p_I, H]
    W_in_p = shard_mlp_col_parallel(W_in, rank, world_size)
    # W_out_p: [H, p_I]
    W_out_p = shard_mlp_row_parallel(W_out, rank, world_size)

    with torch.no_grad():
        expected = mlp(X, W_in, W_out)
        out = tp_mlp(X, W_in_p, W_out_p)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    if should_print:
        print("tensor-parallel MLP correctness check passed.")

    return out


def main() -> None:
    with distributed_context() as (rank, world_size, _LOCAL_RANK, device):
        run_tp_attn(device, rank, world_size, should_print=rank == 0)
        run_tp_mlp(device, rank, world_size, should_print=rank == 0)


if __name__ == "__main__":
    main()
