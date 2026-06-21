#!/usr/bin/env python3
"""Embed Gutenberg passages with LaBSE on GPU -> a memmap .npy for VLDB-scale bench."""

import argparse
import glob
import json
import os
import re
import time

import numpy as np

PG_START = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S
)
PG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)


def passages_from(path):
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        return
    m = PG_START.search(txt)
    if m:
        txt = txt[m.end() :]
    e = PG_END.search(txt)
    if e:
        txt = txt[: e.start()]
    for para in re.split(r"\n\s*\n", txt):
        p = " ".join(para.split())
        if 60 <= len(p) <= 2000:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--glob", default="/archive/gutenberg/texts/*.txt")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(a.model, device="cuda")
    dim = model.get_sentence_embedding_dimension()
    print("model dim", dim, flush=True)
    mm = np.lib.format.open_memmap(
        a.out, mode="w+", dtype=np.float16, shape=(a.target, dim)
    )
    files = sorted(glob.glob(a.glob))
    print("books:", len(files), flush=True)
    buf = []
    n = 0
    t0 = time.time()

    def flush_buf():
        nonlocal n
        if not buf:
            return
        emb = model.encode(
            buf,
            batch_size=a.batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        k = min(len(emb), a.target - n)
        mm[n : n + k] = emb[:k].astype(np.float16)
        n += k
        buf.clear()

    for fi, f in enumerate(files):
        for p in passages_from(f):
            buf.append(p)
        if len(buf) >= 8192:
            flush_buf()
            print(
                f"{n}/{a.target} passages | {n/(time.time()-t0):.0f}/s | book {fi}/{len(files)}",
                flush=True,
            )
            if n >= a.target:
                break
    flush_buf()
    mm.flush()
    json.dump(
        {"n": int(min(n, a.target)), "dim": int(dim), "model": a.model},
        open(a.out + ".meta.json", "w"),
    )
    open(a.out + ".done", "w").write(str(min(n, a.target)))
    print("DONE", min(n, a.target), "vectors ->", a.out, flush=True)


if __name__ == "__main__":
    main()
