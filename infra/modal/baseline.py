"""Modal wrapper for the baseline scripts — this is only a GPU access path.

Everything real lives in scripts/*.sh and runs on any Linux CUDA box without
Modal. This file just builds a container with the vendor fork compiled, mounts
the same scripts, and runs them on an H100.

Usage:
    modal run infra/modal/baseline.py                 # download + bench + trace
    modal run infra/modal/baseline.py --step bench    # just the tok/s table
    modal run infra/modal/baseline.py --step trace    # just the nsys analysis
"""
import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent.parent

app = modal.App("bonsai-turbo-baseline")

data_vol = modal.Volume.from_name("bonsai-turbo-data", create_if_missing=True)

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "cmake", "ninja-build", "build-essential")
    .run_commands(
        # nsys for launch/idle analysis; package name varies by CUDA repo vintage
        "apt-get update && (apt-get install -y cuda-nsight-systems-12-8"
        " || apt-get install -y nsight-systems-cli"
        " || echo 'WARNING: nsys not installed')"
    )
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # scripts are mounted fresh at runtime — editing them never rebuilds the image;
    # the fork itself is compiled once by build_fork (32 CPUs) into the volume
    .add_local_dir(REPO_ROOT / "scripts", "/repo/scripts")
)

hf_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub[hf_transfer]"
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

RUN_ENV = {
    "FORK_DIR": "/data/fork",
    "WEIGHTS_DIR": "/data/weights",
    "OUT_DIR": "/data/results",
    "LD_LIBRARY_PATH": "/data/fork/build/bin",
}


@app.function(image=hf_image, volumes={"/data": data_vol}, timeout=3600)
def download_weights():
    from huggingface_hub import hf_hub_download

    for repo, file in [
        ("prism-ml/Ternary-Bonsai-27B-gguf", "Ternary-Bonsai-27B-Q2_0.gguf"),
        ("prism-ml/Bonsai-27B-gguf", "Bonsai-27B-Q1_0.gguf"),
    ]:
        print(f"fetching {repo} :: {file}")
        hf_hub_download(repo_id=repo, filename=file, local_dir="/data/weights")
    data_vol.commit()
    return "weights ready"


def _run_script(name: str, extra_env: dict | None = None) -> str:
    proc = subprocess.run(
        ["bash", f"/repo/scripts/{name}"],
        env={**__import__("os").environ, **RUN_ENV, **(extra_env or {})},
        capture_output=True, text=True,
    )
    out = proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.returncode else "")
    print(out)
    data_vol.commit()
    if proc.returncode:
        raise RuntimeError(f"{name} exited {proc.returncode}")
    return out


@app.function(image=cuda_image, volumes={"/data": data_vol}, timeout=3600, cpu=32)
def build_fork():
    # compile on local disk (fast), persist only build/bin (binaries + libs)
    out = _run_script("build_vendor_fork.sh", {"FORK_DIR": "/tmp/fork"})
    subprocess.run(
        ["bash", "-c",
         "mkdir -p /data/fork/build && cp -a /tmp/fork/build/bin /data/fork/build/"
         " && cp -a /tmp/fork/include /data/fork/"
         " && mkdir -p /data/fork/ggml && cp -a /tmp/fork/ggml/include /data/fork/ggml/"],
        check=True,
    )
    data_vol.commit()
    return out


@app.function(image=cuda_image, gpu="H100", volumes={"/data": data_vol}, timeout=2400)
def bench():
    return _run_script("bench_baseline.sh")


@app.function(image=cuda_image, gpu="H100", volumes={"/data": data_vol}, timeout=3600)
def trace():
    return _run_script("trace_baseline.sh", {"TRACE_STAGE_LOCAL": "1"})


@app.local_entrypoint()
def main(step: str = "all"):
    if step == "all":
        dl = download_weights.spawn()
        fk = build_fork.spawn()
        print(dl.get())
        print(fk.get())
        print(bench.remote())
        print(trace.remote())
        return
    if step == "download":
        print(download_weights.remote())
    if step == "build":
        print(build_fork.remote())
    if step == "bench":
        print(bench.remote())
    if step == "trace":
        print(trace.remote())
