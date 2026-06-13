from dataclasses import dataclass
import modal

app = modal.App("tp-attn")
trace_volume = modal.Volume.from_name("tp-attn-traces", create_if_missing=True)
TRACE_ROOT = "/traces"
GPU_CONFIG = "h100:8"
NPROC_PER_NODE = 8
BENCH_B = 8
BENCH_S = 65536
BENCH_H = 4096
BENCH_NUM_HEADS = 32

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.13",
    )
    .entrypoint([])
    .uv_pip_install("torch>=2.12.0", "numpy", "matplotlib")
    .add_local_file("profiler_utils.py", remote_path="/root/profiler_utils.py")
    .add_local_file("tensor_parallelism.py", remote_path="/root/tp.py")
    .add_local_file("sequence_parallelism.py", remote_path="/root/sp.py")
    .add_local_file("ts_parallelism.py", remote_path="/root/tsp.py")
)


@dataclass
class Input:
    batch_size: int
    seq_len: int
    hidden_size: int
    num_heads: int


def _run_torchrun(script_path: str, trace_name: str, input: Input) -> list[str]:
    import os
    import subprocess

    trace_dir = f"{TRACE_ROOT}/{trace_name}"
    os.makedirs(trace_dir, exist_ok=True)

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "NCCL_NVLS_ENABLE": "0",
        "TRACE_DIR": trace_dir,
        "EXPECTED_WORLD_SIZE": str(NPROC_PER_NODE),
        "BENCH_B": str(input.batch_size),
        "BENCH_S": str(input.seq_len),
        "BENCH_H": str(input.hidden_size),
        "BENCH_NUM_HEADS": str(input.num_heads),
    }
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={NPROC_PER_NODE}",
            script_path,
        ],
        check=True,
        env=env,
    )

    trace_volume.commit()
    return sorted(
        os.path.relpath(os.path.join(root, filename), TRACE_ROOT)
        for root, _dirs, files in os.walk(trace_dir)
        for filename in files
    )


@app.function(
    gpu=GPU_CONFIG,
    image=image,
    volumes={TRACE_ROOT: trace_volume},
    timeout=30 * 60,
)
def run() -> list[str]:
    inputs = [
        Input(batch_size=8, seq_len=8192, hidden_size=2048, num_heads=16),
        Input(batch_size=8, seq_len=16384, hidden_size=2048, num_heads=16),
        Input(batch_size=8, seq_len=32768, hidden_size=2048, num_heads=16),
        Input(batch_size=8, seq_len=65536, hidden_size=2048, num_heads=16),
        Input(batch_size=8, seq_len=8192, hidden_size=2048, num_heads=32),
        Input(batch_size=8, seq_len=16384, hidden_size=2048, num_heads=32),
        Input(batch_size=8, seq_len=32768, hidden_size=2048, num_heads=32),
        Input(batch_size=8, seq_len=65536, hidden_size=2048, num_heads=32),
        Input(batch_size=8, seq_len=2 * 65536, hidden_size=2048, num_heads=16),
        Input(batch_size=8, seq_len=2 * 65536, hidden_size=2048, num_heads=32),
        Input(batch_size=8, seq_len=4 * 65536, hidden_size=2048, num_heads=16),
        Input(batch_size=8, seq_len=4 * 65536, hidden_size=2048, num_heads=32),
        Input(batch_size=8, seq_len=8 * 65536, hidden_size=2048, num_heads=16),
        Input(batch_size=8, seq_len=8 * 65536, hidden_size=2048, num_heads=32),
    ]
    for input in inputs:
        print("tp", input)
        _run_torchrun("/root/tp.py", "tp", input)
    for input in inputs:
        print("sp", input)
        _run_torchrun("/root/sp.py", "sp", input)
    for input in inputs:
        print("tsp", input)
        _run_torchrun("/root/tsp.py", "tsp", input)
