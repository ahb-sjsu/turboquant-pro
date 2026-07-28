#!/usr/bin/env python3
"""Arm D2: encode the committed seed-7 Ethics sample with LaBSE at max_seq_length=512.

Replication predicate: the SAME rows, in the SAME order, as sample_emb.npy
(BGE-M3) -- sample_ids.npy holds ethics_chunks.id per row, so the texts are
identical by construction. An order-sensitive text fingerprint is recorded so
that is verifiable rather than asserted.

max_seq_length is set to 512, LaBSE's architectural limit
(max_position_embeddings=512, absolute), NOT the sentence-transformers default
of 256. Measured effect (d2_truncation_audit_256_512.json): per-language
truncation spread collapses 0.766 -> 0.117 and the median row of every
scored language is kept whole. This is a DECLARED configuration deviation from
the ST default -- the Gutenberg LaBSE arm was encoded at the 256 default.

Mirrors gpu_embed.py otherwise: SentenceTransformer, device cuda,
batch_size 32, normalize_embeddings=True. GPU 1 only (GPU 0 is Erebus).
"""
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

OUT = "/archive/tqp_strata"
DST = f"{OUT}/ethics_labse.npy"
MODEL = "sentence-transformers/LaBSE"
MAXLEN, BATCH, FETCH = 512, 32, 20_000
GPU_MAX_C = 80.0


def gpu_temp() -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader",
             "--id=" + os.environ.get("CUDA_VISIBLE_DEVICES", "1")],
            capture_output=True, text=True, timeout=20,
        )
        return float(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - guard is advisory
        return 0.0


def cool():
    for _ in range(40):
        t = gpu_temp()
        if t <= 0.0 or t < GPU_MAX_C:
            return
        print(f"  [thermal] GPU {t:.0f}C >= {GPU_MAX_C:.0f}, waiting 30s", flush=True)
        time.sleep(30)


ids = np.load(f"{OUT}/sample_ids.npy")
n = len(ids)
print(f"sample: {n} rows", flush=True)

# ---- fetch texts in the committed ROW ORDER -----------------------------
conn = psycopg2.connect(dbname="atlas", user="claude")
cur = conn.cursor()
pos = {int(v): i for i, v in enumerate(ids)}
texts: list[str | None] = [None] * n
got = 0
t0 = time.time()
for s in range(0, n, FETCH):
    chunk = [int(v) for v in ids[s: s + FETCH]]
    cur.execute("SELECT id, content FROM ethics_chunks WHERE id = ANY(%s)", (chunk,))
    for rid, content in cur.fetchall():
        texts[pos[rid]] = content
        got += 1
    print(f"  fetched {got}/{n} ({time.time() - t0:.0f}s)", flush=True)
conn.close()
if got != n or any(t is None for t in texts):
    raise SystemExit(f"FATAL: fetched {got} of {n}; refusing to encode a partial arm")

fp = hashlib.sha256()
for t in texts:
    fp.update(t.encode("utf-8"))
    fp.update(b"\x00")
text_fp = fp.hexdigest()
print(f"order-sensitive text fingerprint: {text_fp}", flush=True)

# ---- encode ------------------------------------------------------------
cool()
model = SentenceTransformer(MODEL, device="cuda")
model.max_seq_length = MAXLEN
print(f"{MODEL} loaded; max_seq_length={model.max_seq_length}", flush=True)

emb = np.empty((n, model.get_sentence_embedding_dimension()), dtype=np.float32)
CH = 20_000
t0 = time.time()
for s in range(0, n, CH):
    cool()
    part = texts[s: s + CH]
    emb[s: s + len(part)] = model.encode(
        part, batch_size=BATCH, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  encoded {s + len(part)}/{n}  gpu={gpu_temp():.0f}C  "
          f"elapsed={time.time() - t0:.0f}s", flush=True)

np.save(DST, emb)
meta = {
    "arm": "D2",
    "model": MODEL,
    "max_seq_length": MAXLEN,
    "max_seq_length_note": "512 = LaBSE architectural limit; ST default is 256 "
                           "(declared deviation, see d2_truncation_audit_256_512.json)",
    "normalize_embeddings": True,
    "batch_size": BATCH,
    "n": int(n),
    "dim": int(emb.shape[1]),
    "row_order": "sample_ids.npy (ethics_chunks.id), identical to sample_emb.npy",
    "text_fingerprint_sha256": text_fp,
    "counterpart": "sample_emb.npy (BGE-M3, 1024d) — same ids, same order",
    "prereg_blob": "fc91233e3bdba70f42e4d9a98ce1f8b500ccfe0d",
    "scored": False,
    "scored_note": "arm D2 is BUILT but NOT SCORED: the corrected role statistic "
                   "is not yet registered. Scoring before that amendment is frozen "
                   "would spend this arm's blindness.",
}
with open(DST + ".meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"saved {DST} {emb.shape} in {time.time() - t0:.0f}s", flush=True)
with open(f"{OUT}/D2_DONE.flag", "w") as f:
    f.write("ok\n")
