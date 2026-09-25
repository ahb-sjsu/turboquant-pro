"""Behaviour of a quantized variant: KL from the full-precision model on held-out text (Part III).

The reference model and the variant are two copies on one GPU. For each variant the
variant's decoder matrices are rewritten from the reference's weights at the variant's bit
widths (``quant.rtn``), and every evaluation sequence is scored by both. Recorded per
sequence: summed KL(p_ref || p_var) over positions, token count, and top-1 agreement, so the
scorer can form split halves.
"""

from __future__ import annotations

import torch

from .quant import rtn
from .tables import linear_modules


@torch.no_grad()
def apply_variant(ref, var, bits: dict) -> None:
    rm, vm = linear_modules(ref), linear_modules(var)
    for name, b in bits.items():
        vm[name].weight.copy_(rtn(rm[name].weight, b).to(vm[name].weight.dtype))


@torch.no_grad()
def kl_per_sequence(ref, var, seqs: list) -> list:
    out = []
    for ids in seqs:
        lr = torch.log_softmax(ref(ids, use_cache=False).logits.float(), -1)
        lv = torch.log_softmax(var(ids, use_cache=False).logits.float(), -1)
        kl = (lr.exp() * (lr - lv)).sum(-1)  # per position
        agree = (lr.argmax(-1) == lv.argmax(-1)).float()
        out.append(
            {
                "kl_sum": float(kl.sum()),
                "tokens": int(kl.numel()),
                "top1_agree": float(agree.mean()),
            }
        )
        del lr, lv
    return out
