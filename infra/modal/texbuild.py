"""Build docs/assets/overview.png from the LaTeX source, in a texlive container.

Anyone with texlive can build it locally instead:
    pdflatex overview.tex && pdftoppm -png -r 220 -singlefile overview.pdf overview

Usage: modal run infra/modal/texbuild.py
"""
import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent.parent

app = modal.App("bonsai-turbo-texbuild")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("texlive-latex-base", "texlive-latex-extra", "texlive-pictures",
                 "texlive-fonts-recommended", "poppler-utils")
    .add_local_file(REPO_ROOT / "docs" / "assets" / "overview.tex", "/work/overview.tex")
)


@app.function(image=image, timeout=600)
def build() -> bytes:
    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", "/tmp", "/work/overview.tex"],
        ["pdftoppm", "-png", "-scale-to-x", "2560", "-scale-to-y", "1280",
         "-singlefile", "/tmp/overview.pdf", "/tmp/overview"],
    ):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"{cmd[0]} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-1000:]}")
    return Path("/tmp/overview.png").read_bytes()


@app.local_entrypoint()
def main():
    png = build.remote()
    out = REPO_ROOT / "docs" / "assets" / "overview.png"
    out.write_bytes(png)
    print(f"wrote {out} ({len(png)} bytes)")
