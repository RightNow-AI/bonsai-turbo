"""Modal dev loop — build the engine, run unit tests, inspect real weights.

Like baseline.py this is only a GPU/CPU access path: the same build works on
any Linux box via plain cmake. Source is mounted fresh on every run, so
iterating never rebuilds the image.

Usage:
    modal run infra/modal/dev.py                    # build + ctest
    modal run infra/modal/dev.py --inspect ternary  # + dump gguf metadata
    modal run infra/modal/dev.py --inspect ternary --scan  # + code-3 scan
"""
import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent.parent

app = modal.App("bonsai-turbo-dev")

data_vol = modal.Volume.from_name("bonsai-turbo-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++", "cmake", "ninja-build")
    .add_local_dir(
        REPO_ROOT, "/repo",
        ignore=["third_party", ".git", "weights", "results", "build", "__pycache__"],
    )
)

cuda_dev_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("g++", "cmake", "ninja-build")
    .add_local_dir(
        REPO_ROOT, "/repo",
        ignore=["third_party", ".git", "weights", "results", "build", "__pycache__"],
    )
)

MODELS = {
    "ternary": "/data/weights/Ternary-Bonsai-27B-Q2_0.gguf",
    "onebit": "/data/weights/Bonsai-27B-Q1_0.gguf",
}


def _sh(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + (("\n--- stderr ---\n" + proc.stderr) if proc.returncode else "")
    print(out)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(cmd)} exited {proc.returncode}")
    return out


@app.function(image=image, volumes={"/data": data_vol}, timeout=900, cpu=8)
def build_test(inspect: str = "", scan: bool = False) -> str:
    report = []
    _sh(["cmake", "-S", "/repo", "-B", "/tmp/build", "-G", "Ninja"])
    _sh(["cmake", "--build", "/tmp/build", "-j"])
    report.append(_sh(["ctest", "--test-dir", "/tmp/build", "--output-on-failure"]))

    if inspect:
        model = MODELS[inspect]
        if not Path(model).exists():
            report.append(f"!! {model} not in volume yet — run baseline.py download first")
        else:
            args = ["/tmp/build/bt-inspect", model] + (["--scan-code3"] if scan else [])
            out = _sh(args)
            Path("/data/results").mkdir(exist_ok=True)
            Path(f"/data/results/inspect_{inspect}.txt").write_text(out)
            data_vol.commit()
            report.append(out)
    return "\n".join(report)


@app.function(image=cuda_dev_image, gpu="H100", volumes={"/data": data_vol}, timeout=1200)
def microbench(shapes: str = "", gguf: str = "", tensor: str = "") -> str:
    _sh(["cmake", "-S", "/repo", "-B", "/tmp/build", "-G", "Ninja"])
    _sh(["cmake", "--build", "/tmp/build", "-j"])
    _sh(["ctest", "--test-dir", "/tmp/build", "--output-on-failure"])
    args = ["/tmp/build/bt-microbench"]
    if gguf:
        args += ["--gguf", MODELS.get(gguf, gguf), tensor]
    elif shapes:
        args += [shapes]
    return _sh(args)


@app.local_entrypoint()
def main(inspect: str = "", scan: bool = False, gpu: bool = False,
         shapes: str = "", gguf: str = "", tensor: str = ""):
    if gpu:
        print(microbench.remote(shapes=shapes, gguf=gguf, tensor=tensor))
    else:
        print(build_test.remote(inspect=inspect, scan=scan))
