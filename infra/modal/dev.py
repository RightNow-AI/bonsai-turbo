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
    .pip_install("jinja2", "gguf", "requests")
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


@app.function(image=cuda_dev_image, gpu="H100", memory=32768, volumes={"/data": data_vol}, timeout=1200)
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


@app.function(image=cuda_dev_image, gpu="H100", memory=32768, volumes={"/data": data_vol}, timeout=3600)
def parity(n_gen: int = 64, model: str = "ternary", flags: str = "") -> str:
    _sh(["cmake", "-S", "/repo", "-B", "/tmp/build", "-G", "Ninja"])
    _sh(["cmake", "--build", "/tmp/build", "-j"])
    env = {
        "FORK_DIR": "/data/fork",
        "WEIGHTS_DIR": "/data/weights",
        "OUT_DIR": "/tmp/parity_out",
        "BT_BUILD": "/tmp/build",
        "PARITY_MODEL": MODELS[model],
        "N_GEN": str(n_gen),
        "BT_RUN_FLAGS": flags,
        "LD_LIBRARY_PATH": "/data/fork/build/bin",
    }
    import os

    proc = subprocess.run(["bash", "/repo/scripts/parity.sh"],
                          env={**os.environ, **env}, capture_output=True, text=True)
    out = proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.returncode else "")
    print(out)
    Path("/data/results").mkdir(exist_ok=True)
    Path("/data/results/parity_summary.txt").write_text(out[-20000:])
    data_vol.commit()
    if proc.returncode:
        raise RuntimeError(f"parity gate failed ({proc.returncode})")
    return out


@app.function(image=cuda_dev_image, gpu="H100", memory=32768, volumes={"/data": data_vol}, timeout=1800)
def speed(n_gen: int = 128, model: str = "ternary") -> str:
    _sh(["cmake", "-S", "/repo", "-B", "/tmp/build", "-G", "Ninja"])
    _sh(["cmake", "--build", "/tmp/build", "-j"])
    # tg128-comparable: short fixed prompt, time n_gen greedy decode steps
    ids = "785,9426,1614,315,22670,5068,2727,429"
    out = ["== cuda graph =="]
    out.append(_sh(["/tmp/build/bt-run", "--model", MODELS[model],
                    "--ids", ids, "--n", str(n_gen), "--bench", "--graph"]))
    out.append("== megakernel ==")
    out.append(_sh(["/tmp/build/bt-run", "--model", MODELS[model],
                    "--ids", ids, "--n", str(n_gen), "--bench", "--mega"]))
    return "\n".join(out)


@app.function(image=cuda_dev_image, gpu="H100", memory=32768, volumes={"/data": data_vol}, timeout=1800)
def debug_probe(prompt: str = "Hello", model: str = "ternary", level: str = "2",
                flags: str = "") -> str:
    import os
    import re

    _sh(["cmake", "-S", "/repo", "-B", "/tmp/build", "-G", "Ninja"])
    _sh(["cmake", "--build", "/tmp/build", "-j"])
    env = {**os.environ, "LD_LIBRARY_PATH": "/data/fork/build/bin"}

    tok = subprocess.run(["/data/fork/build/bin/llama-tokenize", "-m", MODELS[model],
                          "-p", prompt, "--ids"], env=env, capture_output=True, text=True)
    ids = re.sub(r"[\[\] ]", "", tok.stdout.strip().splitlines()[-1])
    print(f"prompt {prompt!r} -> ids {ids}")

    import shlex

    extra = shlex.split(flags)
    ours = subprocess.run(
        ["/tmp/build/bt-run", "--model", MODELS[model], "--ids", ids, "--n", "4"] + extra,
        env={**env, "BT_PROBE": level}, capture_output=True, text=True)
    print("=== OURS ===")
    print(ours.stderr[-8000:])

    _sh(["bash", "-c",
         "g++ -O2 -I/data/fork/include -I/data/fork/ggml/include "
         "/repo/tools/vendor_probe.cpp -L/data/fork/build/bin -lllama -lggml "
         "-lggml-base -Wl,-rpath,/data/fork/build/bin -o /tmp/vendor-probe"])
    vend = subprocess.run(["/tmp/vendor-probe", MODELS[model], ids], env=env,
                          capture_output=True, text=True)
    print("=== VENDOR (layer 0) ===")
    print("\n".join(l for l in vend.stdout.splitlines() if l.startswith("vprobe")))
    if vend.returncode:
        print("vendor-probe stderr tail:", vend.stderr[-1500:])
    return "done"


@app.function(image=cuda_dev_image, gpu="H100", memory=32768, volumes={"/data": data_vol},
              timeout=3 * 3600, retries=modal.Retries(max_retries=2, initial_delay=10.0))
def math500(n: int = 100, max_gen: int = 8192, model: str = "ternary") -> str:
    import os

    os.environ["LD_LIBRARY_PATH"] = "/data/fork/build/bin"
    _sh(["cmake", "-S", "/repo", "-B", "/tmp/build", "-G", "Ninja"])
    _sh(["cmake", "--build", "/tmp/build", "-j"])
    env = {**os.environ, "LD_LIBRARY_PATH": "/data/fork/build/bin"}
    _sh(["python3", "/repo/scripts/math500.py", "prepare", "--model", MODELS[model],
         "--tokenizer-bin", "/data/fork/build/bin/llama-tokenize",
         "--out", "/tmp/m5", "--n", str(n)])
    with open("/tmp/m5/generated.txt", "w") as out:
        proc = subprocess.run(
            ["/tmp/build/bt-run", "--model", MODELS[model], "--ids-file", "/tmp/m5/ids.txt",
             "--n", str(max_gen), "--ctx", str(max_gen + 2048), "--graph", "--eos", "248046"],
            env=env, stdout=out, stderr=subprocess.PIPE, text=True)
    if proc.returncode:
        print(proc.stderr[-3000:])
        raise RuntimeError("generation failed")
    grade = subprocess.run(
        ["python3", "/repo/scripts/math500.py", "grade", "--model", MODELS[model],
         "--generated", "/tmp/m5/generated.txt", "--refs", "/tmp/m5/refs.json"],
        capture_output=True, text=True)
    print(grade.stdout[-6000:])
    Path("/data/results").mkdir(exist_ok=True)
    Path("/data/results/math500.txt").write_text(grade.stdout)
    data_vol.commit()
    if grade.returncode:
        raise RuntimeError("MATH-500 gate failed")
    return grade.stdout[-2000:]


@app.local_entrypoint()
def main(inspect: str = "", scan: bool = False, gpu: bool = False,
         shapes: str = "", gguf: str = "", tensor: str = "",
         run_parity: bool = False, run_speed: bool = False, run_probe: bool = False,
         run_math: bool = False, n_gen: int = 64, model: str = "ternary",
         flags: str = ""):
    if run_math:
        print(math500.remote(model=model))
    elif run_probe:
        print(debug_probe.remote(model=model, flags=flags))
    elif run_parity:
        print(parity.remote(n_gen=n_gen, model=model, flags=flags))
    elif run_speed:
        print(speed.remote(n_gen=max(n_gen, 128), model=model))
    elif gpu:
        print(microbench.remote(shapes=shapes, gguf=gguf, tensor=tensor))
    else:
        print(build_test.remote(inspect=inspect, scan=scan))
