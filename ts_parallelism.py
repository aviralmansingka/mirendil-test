import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F

from profiler_utils import profile_rank_region
from sequence_parallelism import all_gather_seq, shard_seq_parallel


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("this script requires CUDA")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "launch with: torchrun --standalone --nproc_per_node=8 ts.py"
        )

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=300))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_world_size = os.environ.get("EXPECTED_WORLD_SIZE")
    if expected_world_size is not None and world_size != int(expected_world_size):
        raise RuntimeError(
            f"expected world_size == {expected_world_size}, got {world_size}"
        )

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


def broadcast_W_qkvo_r(
    W_q_p: torch.Tensor,
    W_k_p: torch.Tensor,
    W_v_p: torch.Tensor,
    W_o_p: torch.Tensor,
    owner_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()

    if rank == owner_rank:
        W_q_r = W_q_p
        W_k_r = W_k_p
        W_v_r = W_v_p
        W_o_r = W_o_p
    else:
        W_q_r = torch.empty_like(W_q_p)
        W_k_r = torch.empty_like(W_k_p)
        W_v_r = torch.empty_like(W_v_p)
        W_o_r = torch.empty_like(W_o_p)

    dist.broadcast(W_q_r, src=owner_rank)
    dist.broadcast(W_k_r, src=owner_rank)
    dist.broadcast(W_v_r, src=owner_rank)
    dist.broadcast(W_o_r, src=owner_rank)
    return W_q_r, W_k_r, W_v_r, W_o_r


def broadcast_W_mlp_r(
    W_in_p: torch.Tensor,
    W_out_p: torch.Tensor,
    owner_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()

    if rank == owner_rank:
        W_in_r = W_in_p
        W_out_r = W_out_p
    else:
        W_in_r = torch.empty_like(W_in_p)
        W_out_r = torch.empty_like(W_out_p)

    dist.broadcast(W_in_r, src=owner_rank)
    dist.broadcast(W_out_r, src=owner_rank)
    return W_in_r, W_out_r


def tsp_attn(X_p, W_q_p, W_k_p, W_v_p, W_o_p, num_heads):
    """
    for r: 0...D
        broadcast: W_r
        compute Q_rp, K_rp, V_rp
        all-gather K_r, V_r
        A_rp = causal_attn(Q_rp, K_r, V_r)
        Y_p += A_rp @ Wo_r
    """
    B, local_seq, H = X_p.shape
    rank = dist.get_rank()

    world_size = dist.get_world_size()
    total_seq = local_seq * world_size
    local_hidden = W_q_p.shape[0]
    if num_heads % world_size != 0:
        raise ValueError("num_heads must be divisible by world_size")
    local_heads = num_heads // world_size
    if local_hidden % local_heads != 0:
        raise ValueError("local_hidden must be divisible by local_heads")
    D = local_hidden // local_heads
    out = torch.zeros_like(X_p)
    for r in range(world_size):
        W_q_r, W_k_r, W_v_r, W_o_r = broadcast_W_qkvo_r(
            W_q_p,
            W_k_p,
            W_v_p,
            W_o_p,
            owner_rank=r,
        )
        q_rp = project_to_heads(F.linear(X_p, W_q_r), local_heads, D)
        k_rp = project_to_heads(F.linear(X_p, W_k_r), local_heads, D).contiguous()
        v_rp = project_to_heads(F.linear(X_p, W_v_r), local_heads, D).contiguous()

        # k/v: [B, local_heads, S, D]
        k_r = all_gather_seq(k_rp)
        v_r = all_gather_seq(v_rp)

        q_positions = torch.arange(
            rank * local_seq,
            (rank + 1) * local_seq,
            device=X_p.device,
        )
        k_positions = torch.arange(total_seq, device=X_p.device)
        causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)

        attn_pr = F.scaled_dot_product_attention(
            q_rp, k_r, v_r, attn_mask=causal_mask, dropout_p=0.0
        )
        attn_pr = attn_pr.transpose(1, 2).contiguous().view(B, local_seq, local_hidden)

        out += F.linear(attn_pr, W_o_r)

    return out


def tsp_mlp(
    X_p: torch.Tensor,
    W_in_p: torch.Tensor,
    W_out_p: torch.Tensor,
) -> torch.Tensor:
    _B, _local_seq, H = X_p.shape
    assert_mlp_shapes(H, W_in_p, W_out_p)

    world_size = dist.get_world_size()
    out = torch.zeros_like(X_p)
    for r in range(world_size):
        W_in_r, W_out_r = broadcast_W_mlp_r(W_in_p, W_out_p, owner_rank=r)
        hidden_pr = F.gelu(F.linear(X_p, W_in_r))
        out += F.linear(hidden_pr, W_out_r)

    return out


def assert_tp_config(H: int, num_heads: int, world_size: int) -> None:
    if H % num_heads != 0:
        raise ValueError("H must be divisible by num_heads")
    if num_heads % world_size != 0:
        raise ValueError("num_heads must be divisible by world_size")


def assert_mlp_tsp_config(intermediate_size: int, world_size: int) -> None:
    if intermediate_size % world_size != 0:
        raise ValueError("intermediate_size must be divisible by world_size")


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


def project_to_heads(
    projected: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    batch_size, seq_len, _hidden_size = projected.shape
    return projected.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


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


def broadcast_from_rank0(tensor: torch.Tensor) -> torch.Tensor:
    dist.broadcast(tensor, src=0)
    return tensor


def init_tensor(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    fan_in = shape[-1]
    scale = fan_in**-0.5
    return torch.randn(*shape, device=device) * scale


def run_tsp_attn(device: torch.device, rank: int, world_size: int, should_print: bool):
    B = int(os.environ.get("BENCH_B", "32"))
    S = int(os.environ.get("BENCH_S", "65536"))
    H = int(os.environ.get("BENCH_H", "4096"))
    num_heads = int(os.environ.get("BENCH_NUM_HEADS", "32"))
    if H % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")

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

    X_p = shard_seq_parallel(X, rank, world_size)

    # W_q_p/W_k_p/W_v_local: [local_hidden, H]
    W_q_p = shard_col_parallel(W_q, rank, world_size, num_heads)
    W_k_p = shard_col_parallel(W_k, rank, world_size, num_heads)
    W_v_p = shard_col_parallel(W_v, rank, world_size, num_heads)
    # W_o_p: [H, p_hidden]
    W_o_p = shard_row_parallel(W_o, rank, world_size, num_heads)

    with torch.no_grad():
        expected = attn(X, W_q, W_k, W_v, W_o, num_heads)
        _ = tsp_attn(X_p, W_q_p, W_k_p, W_v_p, W_o_p, num_heads)
        torch.cuda.synchronize(device)

        with profile_rank_region("tsp_attention", device):
            out_p = tsp_attn(X_p, W_q_p, W_k_p, W_v_p, W_o_p, num_heads)

    expected_p = shard_seq_parallel(expected, rank, world_size)
    torch.testing.assert_close(out_p, expected_p, rtol=1e-5, atol=1e-5)

    gathered = [torch.empty_like(out_p) for _ in range(world_size)]
    dist.all_gather(gathered, out_p)
    out = torch.cat(gathered, dim=1)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    if should_print:
        print("tensor-sequence-parallel causal attention correctness check passed.")

    return out


def shard_mlp_col_parallel(
    W: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    I, _H = W.shape
    assert_mlp_tsp_config(I, world_size)
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
    assert_mlp_tsp_config(I, world_size)
    p_I = I // world_size
    start = rank * p_I
    end = start + p_I
    return W[:, start:end].contiguous()


def run_tsp_mlp(device: torch.device, rank: int, world_size: int, should_print: bool):
    B = int(os.environ.get("BENCH_B", "32"))
    S = int(os.environ.get("BENCH_S", "65536"))
    H = int(os.environ.get("BENCH_H", "4096"))
    I = int(os.environ.get("BENCH_I", "8192"))
    assert_mlp_tsp_config(I, world_size)

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

    X_p = shard_seq_parallel(X, rank, world_size)

    with torch.no_grad():
        expected = mlp(X, W_in, W_out)
        _ = tsp_mlp(X_p, W_in_p, W_out_p)
        torch.cuda.synchronize(device)

        with profile_rank_region("tsp_mlp", device):
            out_p = tsp_mlp(X_p, W_in_p, W_out_p)

    expected_p = shard_seq_parallel(expected, rank, world_size)
    torch.testing.assert_close(out_p, expected_p, rtol=1e-5, atol=1e-5)

    gathered = [torch.empty_like(out_p) for _ in range(world_size)]
    dist.all_gather(gathered, out_p)
    out = torch.cat(gathered, dim=1)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    if should_print:
        print("tensor-sequence-parallel MLP correctness check passed.")

    return out_p


def main():
    with distributed_context() as (rank, world_size, _LOCAL_RANK, device):
        run_tsp_attn(device, rank, world_size, should_print=rank == 0)
        run_tsp_mlp(device, rank, world_size, should_print=rank == 0)


if __name__ == "__main__":
    main()
