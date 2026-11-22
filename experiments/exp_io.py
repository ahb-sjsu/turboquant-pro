# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Model and input loading shared by the matched-bit experiments E1 to E4.

By default a model comes from Hugging Face (from_pretrained, then moved to the
device) and the WikiText-2 sequences come from `datasets` through the model's
tokenizer, as on Atlas. On an NRP pod, environment variables switch both to staged
copies, so the pod installs, downloads and tokenizes nothing:

  NB_MODEL_DIR   the model's config directory
  NB_STRIPES     a glob of weight stripes, read straight into the device
                 (nrp_tmlr/loader.py)
  NB_TOKENS      the same sequences, tokenized on Atlas (nrp_tmlr/driver.py tokens)
  NB_THREADS     CPU threads when the thermal controller is absent (default 4)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nrp_tmlr"))


def load_texts(split: str, n: int) -> list[str]:
    """The first n chunks of over 1200 characters of a WikiText-2 split (E1 to E4)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    out, buf = [], ""
    for row in ds:
        t = row["text"].strip()
        if not t:
            continue
        buf = (buf + " " + t).strip()
        if len(buf) > 1200:
            out.append(buf)
            buf = ""
        if len(out) >= n:
            break
    return out


def fallback_threads() -> int:
    return int(os.environ.get("NB_THREADS", "4"))


def model_and_tokenizer(model_name: str, dtype: torch.dtype, device: str):
    """(model in eval mode on device, tokenizer; None if inputs are pre-tokenized)."""
    stripes = os.environ.get("NB_STRIPES")
    if stripes:
        from loader import load

        files = sorted(glob.glob(stripes))
        if not files:
            raise SystemExit(f"no weight stripes match {stripes}")
        model = load(os.environ["NB_MODEL_DIR"], files, device, dtype)
        tok = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(device).eval()
    return model, tok


def token_batch(tok, split: str, n: int, n_seq: int) -> list[torch.Tensor]:
    """Token ids of a split's first n chunks, truncated to n_seq, 16 or more long."""
    path = os.environ.get("NB_TOKENS")
    if path:
        d = json.load(open(path))
        if d["window"] != n_seq:
            raise SystemExit(f"{path}: window {d['window']}, run asks {n_seq}")
        chunks = d["splits"][split]
        if len(chunks) < n:
            raise SystemExit(f"{path}: {len(chunks)} {split} chunks, run asks {n}")
        ids_list = [torch.tensor(c, dtype=torch.long) for c in chunks[:n]]
    else:
        ids_list = [
            tok(t, return_tensors="pt", truncation=True, max_length=n_seq).input_ids[0]
            for t in load_texts(split, n)
        ]
    return [ids for ids in ids_list if ids.numel() >= 16]
