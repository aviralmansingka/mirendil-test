import modal

app = modal.App("tp-attn")
trace_volume = modal.Volume.from_name("tp-attn-traces", create_if_missing=True)
TRACE_ROOT = "/traces"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.13",
    )
    .entrypoint([])
    .uv_pip_install("torch>=2.12.0")
    .add_local_file("profiler_utils.py", remote_path="/root/profiler_utils.py")
    .add_local_file("tensor_parallelism.py", remote_path="/root/tp.py")
    .add_local_file("sequence_parallelism.py", remote_path="/root/sp.py")
    .add_local_file("sequence_parallelism.py", remote_path="/root/sequence_parallelism.py")
    .add_local_file("ts_parallelism.py", remote_path="/root/ts.py")
)


def _run_torchrun(script_path: str, trace_name: str) -> list[str]:
    import os
    import subprocess

    trace_dir = f"{TRACE_ROOT}/{trace_name}"
    os.makedirs(trace_dir, exist_ok=True)

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "TRACE_DIR": trace_dir,
    }
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
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


@app.function(gpu="h100:2", image=image, volumes={TRACE_ROOT: trace_volume})
def tp() -> list[str]:
    return _run_torchrun("/root/tp.py", "tp")


@app.function(gpu="h100:2", image=image, volumes={TRACE_ROOT: trace_volume})
def sp() -> list[str]:
    return _run_torchrun("/root/sp.py", "sp")


@app.function(gpu="h100:2", image=image, volumes={TRACE_ROOT: trace_volume})
def ts() -> list[str]:
    return _run_torchrun("/root/ts.py", "tsp")


@app.local_entrypoint()
def main():
    for name, fn in (("tp", tp), ("sp", sp), ("tsp", ts)):
        files = fn.remote()
        print(f"{name} traces written to Modal Volume tp-attn-traces:")
        for file in files:
            print(f"  {file}")
