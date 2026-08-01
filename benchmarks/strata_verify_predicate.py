#!/usr/bin/env python3
"""Verify the D1/D2 replication predicate EMPIRICALLY, not by construction.

D2 encodes ethics_chunks.content for the ids in sample_ids.npy. D1's vectors
(sample_emb.npy) came from the DB `embedding` column, written months earlier by
gpu_embed.py from the same column. If any row's `content` had been edited after
its embedding was computed, the predicate ("same texts, different encoder")
would be silently false and no fingerprint taken today could detect it.

Test: re-encode a random sample of rows with BGE-M3 under the original
invocation and compare against the STORED embedding. Cosine ~1.0 means the text
still matches what was embedded, so the texts D2 just saw are the texts D1 was
built from.
"""

import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

SUB = 200
ids = np.load("/archive/tqp_strata/sample_ids.npy")
rng = np.random.default_rng(11)
pick = [int(v) for v in rng.choice(ids, size=SUB, replace=False)]

conn = psycopg2.connect(dbname="atlas", user="claude")
cur = conn.cursor()
cur.execute(
    "SELECT id, content, embedding::float4[] FROM ethics_chunks WHERE id = ANY(%s)",
    (pick,),
)
rows = cur.fetchall()
conn.close()
print(f"fetched {len(rows)} rows", flush=True)

texts = [r[1] for r in rows]
stored = np.array([r[2] for r in rows], dtype=np.float32)

m = SentenceTransformer("BAAI/bge-m3", device="cuda")
fresh = m.encode(
    texts,
    batch_size=8,
    show_progress_bar=False,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

sn = stored / np.maximum(np.linalg.norm(stored, axis=1, keepdims=True), 1e-12)
cos = np.einsum("ij,ij->i", sn, fresh.astype(np.float32))
print(
    f"cosine(stored, re-encoded): min={cos.min():.6f} "
    f"p01={np.percentile(cos, 1):.6f} median={np.median(cos):.6f} "
    f"mean={cos.mean():.6f}"
)
bad = int((cos < 0.99).sum())
print(f"rows below 0.99: {bad}/{len(cos)}")
print(
    "PREDICATE VERIFIED" if bad == 0 else "PREDICATE SUSPECT — content may have changed"
)
