# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Weight stripes: a checkpoint re-laid into K safetensors files of near-equal size.

A GPU pod reads K files on K volumes at once (nrp-kit's volume-set layout), so its
weights arrive in seconds. A plan, made from the cached checkpoint's headers, fixes
every tensor's stripe, name, dtype and offset, so stripes written in a pod are
byte-identical to stripes written on Atlas from the same checkpoint, and their
sha256 can be checked.

    python stripe.py plan --src SNAPSHOT_DIR --k 4 --dtype F16 --out plan.json \
        [--rename gemma3-text]
    python stripe.py write --plan plan.json --src-file FILE --dests D0,D1,D2,D3
    python stripe.py hash --plan plan.json --dests D0,D1,D2,D3

Renaming "gemma3-text" keeps only the language model of Gemma 3 and gives it the key
names of Gemma3ForCausalLM (the matched-bit study perturbs the language model only).
A dtype change is elementwise (bf16 to f16, as from_pretrained(dtype=float16)
casts), done in 64 MB pieces, so a small pod can stripe files many times its memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct

CHUNK = 64 << 20
DONTNEED = getattr(
    os, "POSIX_FADV_DONTNEED", None
)  # drop written pages: bounded page cache
ITEM = {"F16": 2, "BF16": 2, "F32": 4}


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def rename(name: str, rule: str | None) -> str | None:
    if rule is None:
        return name
    if rule == "gemma3-text":
        pre = "language_model.model."
        return "model." + name[len(pre) :] if name.startswith(pre) else None
    raise SystemExit(f"unknown rename rule {rule}")


def make_plan(src: str, k: int, dtype: str | None, rule: str | None) -> dict:
    files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    items = []
    for fn in files:
        header, _ = read_header(os.path.join(src, fn))
        for name, m in header.items():
            if name == "__metadata__":
                continue
            dst = rename(name, rule)
            if dst is None:
                continue
            ddt = dtype or m["dtype"]
            numel = 1
            for s in m["shape"]:
                numel *= s
            items.append(
                {
                    "dst": dst,
                    "src_file": fn,
                    "src": name,
                    "src_dtype": m["dtype"],
                    "dtype": ddt,
                    "shape": m["shape"],
                    "nbytes": numel * ITEM[ddt],
                }
            )
    # greedy balance: largest first, onto the lightest stripe (lowest index on ties)
    loads = [0] * k
    groups: list[list[dict]] = [[] for _ in range(k)]
    for it in sorted(items, key=lambda x: (-x["nbytes"], x["dst"])):
        j = min(range(k), key=lambda s: (loads[s], s))
        groups[j].append(it)
        loads[j] += it["nbytes"]
    stripes = []
    for j, g in enumerate(groups):
        g.sort(key=lambda x: x["dst"])
        header, off = {"__metadata__": {"format": "pt"}}, 0
        for it in g:
            header[it["dst"]] = {
                "dtype": it["dtype"],
                "shape": it["shape"],
                "data_offsets": [off, off + it["nbytes"]],
            }
            it["offset"] = off
            off += it["nbytes"]
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        stripes.append(
            {
                "file": f"model-{j + 1:05d}-of-{k:05d}.safetensors",
                "header": hb.decode(),
                "data_start": 8 + len(hb),
                "size": 8 + len(hb) + off,
                "tensors": g,
            }
        )
    return {"src_files": files, "k": k, "rename": rule, "stripes": stripes}


def _open_stripe(plan_s: dict, dest: str):
    path = os.path.join(dest, plan_s["file"])
    if not os.path.exists(path) or os.path.getsize(path) != plan_s["size"]:
        os.makedirs(dest, exist_ok=True)
        with open(path, "wb") as f:
            hb = plan_s["header"].encode()
            f.write(struct.pack("<Q", len(hb)) + hb)
            f.truncate(plan_s["size"])
    return open(path, "r+b")


def _convert(buf: bytes, src_dtype: str, dtype: str) -> bytes:
    if src_dtype == dtype:
        return buf
    import torch

    tt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    t = torch.frombuffer(bytearray(buf), dtype=tt[src_dtype])
    return t.to(tt[dtype]).numpy().tobytes()


def write(plan: dict, src_file: str, dests: list[str]) -> int:
    """Copy every tensor of one source file into its stripe. Returns bytes written."""
    fn = os.path.basename(src_file)
    if fn not in plan["src_files"]:
        raise SystemExit(f"{fn} is not a source file of this plan")
    sh, base = read_header(src_file)
    total = 0
    with open(src_file, "rb") as fi:
        for j, ps in enumerate(plan["stripes"]):
            mine = [t for t in ps["tensors"] if t["src_file"] == fn]
            if not mine:
                continue
            with _open_stripe(ps, dests[j]) as fo:
                for t in mine:
                    s0, s1 = sh[t["src"]]["data_offsets"]
                    si, di = ITEM[t["src_dtype"]], ITEM[t["dtype"]]
                    step = CHUNK - CHUNK % si
                    pos = 0
                    while pos < s1 - s0:
                        n = min(step, s1 - s0 - pos)
                        fi.seek(base + s0 + pos)
                        buf = fi.read(n)
                        if len(buf) != n:
                            raise SystemExit(f"{fn}: short read in {t['src']}")
                        out = _convert(buf, t["src_dtype"], t["dtype"])
                        o = ps["data_start"] + t["offset"] + pos // si * di
                        fo.seek(o)
                        fo.write(out)
                        fo.flush()
                        os.fsync(fo.fileno())
                        if DONTNEED is not None:
                            os.posix_fadvise(fo.fileno(), o, len(out), DONTNEED)
                            os.posix_fadvise(fi.fileno(), base + s0 + pos, n, DONTNEED)
                        total += len(out)
                        pos += n
    return total


def stripe_hashes(plan: dict, dests: list[str]) -> list[str]:
    out = []
    for ps, d in zip(plan["stripes"], dests):
        h = hashlib.sha256()
        with open(os.path.join(d, ps["file"]), "rb") as f:
            pos = 0
            while True:
                b = f.read(CHUNK)
                if not b:
                    break
                h.update(b)
                if DONTNEED is not None:
                    os.posix_fadvise(f.fileno(), pos, len(b), DONTNEED)
                pos += len(b)
        out.append(h.hexdigest())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--src", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--dtype", default=None)
    p.add_argument("--rename", default=None)
    p.add_argument("--out", required=True)
    w = sub.add_parser("write")
    w.add_argument("--plan", required=True)
    w.add_argument("--src-file", required=True)
    w.add_argument("--dests", required=True)
    h = sub.add_parser("hash")
    h.add_argument("--plan", required=True)
    h.add_argument("--dests", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan = make_plan(a.src, a.k, a.dtype, a.rename)
        json.dump(plan, open(a.out, "w"), indent=1)
        print(a.out, [s["size"] for s in plan["stripes"]])
    elif a.cmd == "write":
        plan = json.load(open(a.plan))
        n = write(plan, a.src_file, a.dests.split(","))
        print(f"{os.path.basename(a.src_file)}: {n / 1e9:.2f} GB written")
    else:
        plan = json.load(open(a.plan))
        for ps, hx in zip(plan["stripes"], stripe_hashes(plan, a.dests.split(","))):
            print(hx, ps["file"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
