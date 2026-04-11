"""Single-model benchmark helper. Called by benchmark_multi_model.py."""
import gc
import json
import sys

import numpy as np
from datasets import load_dataset

from turboquant_pro.learned_codebook import fit_codebook
from turboquant_pro.pca import PCAMatryoshka
from turboquant_pro.pgvector import TurboQuantPGVector

model_name, pca_dim = sys.argv[1], int(sys.argv[2])

ds = load_dataset("mteb/stsbenchmark-sts")
sents = set()
for s in ["train", "validation", "test"]:
    for r in ds[s]:
        sents.add(r["sentence1"])
        sents.add(r["sentence2"])
max_n = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
corpus = sorted(sents)[:max_n]

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(model_name, trust_remote_code=True)
dim = model.get_embedding_dimension()
embs = np.array(
    model.encode(corpus, show_progress_bar=False, batch_size=64), dtype=np.float32
)
del model
gc.collect()

n = len(embs)
n_train = int(n * 0.8)
train, test = embs[:n_train], embs[n_train:]
nt = len(test)
ep = min(pca_dim, dim - 1, n_train - 1)

# Naive truncation
naive_s = []
for i in range(nt):
    a = test[i]
    b = np.zeros(dim, np.float32)
    b[:ep] = test[i, :ep]
    naive_s.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))

# PCA truncation
pca = PCAMatryoshka(input_dim=dim, output_dim=ep)
pca.fit(train)
pca_s = []
for e in test:
    r = pca.inverse_transform(pca.transform(e))
    pca_s.append(float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30)))

# TQ3
tq = TurboQuantPGVector(dim=dim, bits=3, seed=42)
tq_s = []
for e in test:
    r = tq.decompress_embedding(tq.compress_embedding(e))
    tq_s.append(float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30)))

# Learned codebook
lq = fit_codebook(train, dim=dim, bits=3, seed=42)
lq_s = []
for e in test:
    r = lq.decompress_embedding(lq.compress_embedding(e))
    lq_s.append(float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30)))

# PCA + TQ3
pipe = pca.with_quantizer(bits=3, seed=42)
p_s = []
for e in test:
    r = pipe.decompress(pipe.compress(e))
    p_s.append(float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30)))

print(
    json.dumps(
        {
            "model": model_name,
            "dim": dim,
            "pca": ep,
            "n": n,
            "nt": nt,
            "naive": round(float(np.mean(naive_s)), 4),
            "pca_trunc": round(float(np.mean(pca_s)), 4),
            "tq3": round(float(np.mean(tq_s)), 4),
            "learned": round(float(np.mean(lq_s)), 4),
            "pca_tq3": round(float(np.mean(p_s)), 4),
        }
    )
)
