# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Embed WikiText-2 passages with BGE-M3 (public model + public text) for the
public-data replications (heat_taper_public.py / hubness_public.py).

Run on a GPU host:  python benchmarks/embed_wikitext_bge.py
Writes bge_wikitext.npz (set OUT env to change; ~26 MB, not committed).

Dense embedding = CLS token of the last hidden state (BGE-M3 convention),
saved UNNORMALIZED float32 so norms/density structure are preserved.
"""

import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

N_PASSAGES, WORDS = 6500, 100
OUT = os.environ.get("OUT", "bge_wikitext.npz")

from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
words = " ".join(t for t in ds["text"] if t.strip()).split()
passages = [" ".join(words[i * WORDS : (i + 1) * WORDS]) for i in range(N_PASSAGES)]
print("passages:", len(passages), flush=True)

tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = (
    AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).cuda().eval()
)

embs = []
with torch.no_grad():
    for i in range(0, len(passages), 64):
        batch = tok(
            passages[i : i + 64],
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt",
        ).to("cuda")
        out = model(**batch).last_hidden_state[:, 0]  # CLS, unnormalized
        embs.append(out.float().cpu().numpy())
        if i % 1024 == 0:
            print("embedded", i, flush=True)
x = np.concatenate(embs).astype(np.float32)
np.savez_compressed(OUT, embeddings=x)
print("saved", x.shape, "to", OUT, flush=True)
print("EMBED_DONE", flush=True)
