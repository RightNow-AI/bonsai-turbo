#!/usr/bin/env python3
"""MATH-500 (first-100) harness for bonsai-turbo.

prepare: download the dataset, render prompts with the model's own chat
         template, tokenize with the vendor fork's llama-tokenize, and write
         ids.txt + refs.json.
grade:   detokenize bt-run's generated ids, extract \\boxed{...}, score vs
         references. Gate: within 1.0 point of the vendor's reported ternary
         MATH-500 score (99.20).

Deps: pip install jinja2 gguf requests   (no cloud dependency)
Usage:
  python3 math500.py prepare --model M.gguf --tokenizer-bin llama-tokenize --out DIR [--n 100]
  python3 math500.py grade --model M.gguf --generated DIR/generated.txt --refs DIR/refs.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DATA_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
VENDOR_TERNARY_SCORE = 99.20


def read_gguf_strings(model: str):
    """Minimal GGUF v3 metadata reader (the pip `gguf` package rejects the
    vendor's custom tensor types; we never touch the tensor table anyway)."""
    import struct

    f = open(model, "rb")

    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def s():
        return f.read(u64()).decode("utf-8", errors="replace")

    SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def value(t):
        if t == 8:
            return s()
        if t == 9:
            et, n = u32(), u64()
            return [value(et) for _ in range(n)]
        f.read(SCALAR[t])  # scalar payloads are irrelevant here
        return None

    assert f.read(4) == b"GGUF" and u32() == 3
    u64()  # n_tensors
    n_kv = u64()
    template, tokens = None, None
    for _ in range(n_kv):
        key = s()
        t = u32()
        if key == "tokenizer.chat_template" and t == 8:
            template = s()
        elif key == "tokenizer.ggml.tokens" and t == 9:
            tokens = value(9)
        else:
            value(t)
        if template is not None and tokens is not None:
            break
    f.close()
    return template, tokens


def bytes_to_unicode():
    # gpt2 byte<->unicode table (tokens store bytes as printable unicode)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip([chr(c) for c in cs], bs))


def detok(ids, tokens):
    inv = bytes_to_unicode()
    text_bytes = bytearray()
    for i in ids:
        if 0 <= i < len(tokens):
            for ch in tokens[i]:
                if ch in inv:
                    text_bytes.append(inv[ch])
                else:
                    text_bytes.extend(ch.encode("utf-8"))
    return text_bytes.decode("utf-8", errors="replace")


def extract_boxed(text: str):
    # last NON-EMPTY \boxed{...}
    idx = len(text)
    while True:
        idx = text.rfind("\\boxed{", 0, idx)
        if idx < 0:
            return None
        depth = 0
        start = idx + len("\\boxed{")
        for j in range(start - 1, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    inner = text[start:j]
                    if inner.strip():
                        return inner
                    break
        # empty or unterminated: keep scanning backwards


def as_number(a: str):
    m = re.fullmatch(r"-?\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    if m:
        v = int(m.group(1)) / int(m.group(2))
        return -v if a.startswith("-\\") else v
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", a)
    if m:
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(a.replace(",", ""))
    except ValueError:
        return None


def normalize(ans: str) -> str:
    if ans is None:
        return "<none>"
    a = ans.strip().lower()
    a = a.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac")
    a = a.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    a = a.replace("^\\circ", "").replace("^{\\circ}", "")
    a = re.sub(r"\\text\{[^}]*\}", "", a)  # unit annotations
    a = re.sub(r"\\mbox\{[^}]*\}", "", a)
    a = re.sub(r"\s+", "", a)
    a = re.sub(r"^[a-z]=", "", a)          # "x=5" -> "5"
    a = re.sub(r"\\sqrt\{(\w)\}", r"\\sqrt\1", a)  # \sqrt{2} -> \sqrt2
    if re.fullmatch(r"-?\d+\.0+", a):
        a = a.split(".")[0]
    return a


def answers_match(got: str, want: str) -> bool:
    if got == want:
        return True
    ng, nw = as_number(got), as_number(want)
    return ng is not None and nw is not None and abs(ng - nw) < 1e-9


def cmd_prepare(args):
    import requests

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in requests.get(DATA_URL, timeout=120).text.splitlines() if l.strip()]
    rows = rows[: args.n]

    template, _ = read_gguf_strings(args.model)
    from jinja2 import Template

    tpl = Template(template)

    def tokenize(text: str) -> str:
        last = None
        for extra in (["--parse-special"], []):
            last = subprocess.run(
                [args.tokenizer_bin, "-m", args.model, "-p", text, "--ids"] + extra,
                capture_output=True, text=True)
            lines = [l for l in last.stdout.splitlines() if l.strip().startswith("[")]
            if last.returncode == 0 and lines:
                return re.sub(r"[\[\] ]", "", lines[-1].strip())
        raise SystemExit(
            f"tokenize failed rc={last.returncode}\n--- stdout tail ---\n"
            f"{last.stdout[-500:]}\n--- stderr tail ---\n{last.stderr[-800:]}")

    ids_lines, refs = [], []
    for row in rows:
        msg = row["problem"] + "\n" + INSTRUCTION
        rendered = tpl.render(messages=[{"role": "user", "content": msg}],
                              add_generation_prompt=True)
        ids_lines.append(tokenize(rendered))
        refs.append({"answer": row["answer"], "problem": row["problem"]})

    (out / "ids.txt").write_text("\n".join(ids_lines) + "\n")
    (out / "refs.json").write_text(json.dumps(refs))
    print(f"prepared {len(rows)} problems -> {out}")


def cmd_grade(args):
    _, tokens = read_gguf_strings(args.model)
    refs = json.loads(Path(args.refs).read_text())
    gens = [l.split(":", 1)[1].split() for l in Path(args.generated).read_text().splitlines()
            if l.startswith("generated:")]

    correct = 0
    for i, ref in enumerate(refs):
        if i >= len(gens):
            break
        text = detok([int(t) for t in gens[i]], tokens)
        got = normalize(extract_boxed(text))
        want = normalize(ref["answer"])
        ok = answers_match(got, want)
        correct += ok
        if not ok:
            print(f"  #{i} MISS: got {got!r} want {want!r}")
    n = min(len(refs), len(gens))
    score = 100.0 * correct / max(n, 1)
    print(f"MATH-500[{n}]: {correct}/{n} = {score:.2f} "
          f"(vendor ternary {VENDOR_TERNARY_SCORE}, gate >= {VENDOR_TERNARY_SCORE - 1.0:.2f})")
    sys.exit(0 if score >= VENDOR_TERNARY_SCORE - 1.0 else 1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--model", required=True)
    pp.add_argument("--tokenizer-bin", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--n", type=int, default=100)
    pp.set_defaults(fn=cmd_prepare)
    pg = sub.add_parser("grade")
    pg.add_argument("--model", required=True)
    pg.add_argument("--generated", required=True)
    pg.add_argument("--refs", required=True)
    pg.set_defaults(fn=cmd_grade)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
