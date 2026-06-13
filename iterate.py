import modal

app = modal.App("tp-attn")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.13",
    )
    .entrypoint([])
    .uv_pip_install("torch>=2.12.0")
    .add_local_file("tensor_parallelism.py", remote_path="/root/tp.py")
    .add_local_file("sequence_parallelism.py", remote_path="/root/sp.py")
)


@app.function(gpu="h100:2", image=image)
def tp():
    import subprocess

    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "/root/tp.py",
        ],
        check=True,
    )


@app.function(gpu="h100:2", image=image)
def sp():
    import os
    import subprocess

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
    }
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "/root/sp.py",
        ],
        check=True,
        env=env,
    )
