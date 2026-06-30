#!/usr/bin/env python3
"""Paper experiment runner: KV-key quantization quality across models/tasks/methods.

Unlike ``tq_enh_lb_shard.py`` (a faithful but slow simulation that re-quantizes the
settled window every decode step), this implements the **deployable** method:
**quantize the prefill KV once** and freeze it.

  * On the first cache update of each layer (= the prefill, the long prompt), the
    settled region ``[0 : T-HOT]`` is quantized one time: group-wise per-channel keys
    (NF4 or uniform) with the first ``SINK`` tokens kept fp16 and the top-``OUT_FRAC``
    magnitude entries **per channel** (global over the prefill) kept fp16.
  * The most recent ``HOT`` prefill tokens and **all generated tokens** stay fp16.

For LongBench (max_new_tokens <= HOT) generated tokens never leave the hot window, so
the cache quantizes exactly once per layer -> ~fp16 speed, and quality matches the
slow per-step simulation (verified on Llama-2-7B). Values use per-token uniform.

Config via env; sharded by SHARD_ID / NUM_SHARDS. Model via MODEL (HF id) and
MODEL_KEY (LongBench config key, e.g. ``llama2-7b-chat-4k``) for max-length/prompts.
"""
import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL = os.environ.get("MODEL", "NousResearch/Llama-2-7b-chat-hf")
MODEL_KEY = os.environ.get("MODEL_KEY", "llama2-7b-chat-4k")
LBROOT = os.environ.get("LBROOT", "/root/LongBench/LongBench")
DATADIR = os.environ.get("DATADIR", "/root/lb_data/data")
DATASETS = os.environ.get("DATASETS", "trec,triviaqa,qasper").split(",")

KB = int(os.environ.get("KEY_BITS", "4"))
VB = int(os.environ.get("VAL_BITS", "4"))
G = int(os.environ.get("GROUP", "32"))
HOT = int(os.environ.get("HOT", "128"))  # fp16 recent window (== RESID)
SINK = int(os.environ.get("SINK", "0"))
OUT_FRAC = float(os.environ.get("OUTLIER_FRAC", "0.0"))
NUQ = int(os.environ.get("NUQ", "0"))
NOQUANT = int(os.environ.get("NOQUANT", "0"))
# Tier-1 levers:
#  CODEBOOK: uniform | nf4 | nf4a | quantile | kmeans | kvquant | kivi  (key codebook)
#    quantile/kmeans = ONLINE per-channel codebook fit to THIS prefill (calibration-free,
#    data-optimal -- the both-worlds answer to KVQuant's offline Fisher K-means).
#    kvquant = faithful KVQuant baseline (Hooper et al. 2024): per-channel NON-UNIFORM
#      quantization (NUQ) using OFFLINE Fisher-weighted k-means centroids loaded from a
#      calibration cache (see calibrate_kvquant) + dense-sparse fp16 outliers (OUT_FRAC).
#    kivi = KIVI baseline (Liu et al. 2024): per-channel ASYMMETRIC (min/max zero-point)
#      uniform keys at KEY_BITS (default 2) + per-token asymmetric uniform values + an
#      fp16 residual window of the most recent tokens (== the existing HOT window).
#  PREROPE: quantize keys in pre-RoPE space (RoPE smears per-channel structure).
CODEBOOK = os.environ.get("CODEBOOK", "nf4" if NUQ else "uniform")
PREROPE = int(os.environ.get("PREROPE", "0"))
KMEANS_ITERS = int(os.environ.get("KMEANS_ITERS", "8"))
# KVQuant offline-calibration knobs: number of calibration sequences and the on-disk
# cache path the eval run loads centroids from (written by calibrate_kvquant).
KVQ_CALIB_N = int(os.environ.get("KVQ_CALIB_N", "16"))
KVQ_CACHE = os.environ.get("KVQ_CACHE", os.path.join(os.getcwd(), "kvq_calib.pt"))
# KIVI keys default to 2-bit unless KEY_BITS is set explicitly.
if CODEBOOK == "kivi" and "KEY_BITS" not in os.environ:
    KB = 2
TAG = os.environ.get("TAG", "v0")
SHARD = int(os.environ["SHARD_ID"])
NSH = int(os.environ["NUM_SHARDS"])
CHAT = int(os.environ.get("CHAT", "1"))  # wrap prompt in the model chat template
_MAXGEN = int(os.environ.get("MAXGEN", "0"))  # >0 overrides per-task max_new_tokens (ablation)
OUT = f"/root/out_{TAG}"
os.makedirs(OUT, exist_ok=True)

NF4 = torch.tensor(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ]
)

# LongBench model max-context (truncation) and the chat-wrap style per model family.
MODEL_MAXLEN = {
    "llama2-7b-chat-4k": 3500,
    "llama2-13b-chat-4k": 3500,
    "mistral-7b-instruct": 31500,
    "qwen2.5-7b-instruct": 31500,
}
MAXLEN = int(os.environ.get("MAXLEN", str(MODEL_MAXLEN.get(MODEL_KEY, 3500))))

# Layer index threaded out of the cache-update hook (_patched_update sets this to `li`
# before it calls qdq_key_block) so the per-layer KVQuant calibrated codebook can be
# looked up inside qdq_key_block. Default 0 keeps non-kvquant paths unaffected.
_CUR_LAYER = 0
# Prefill layer counter for the pre-RoPE hook (the shared apply_rotary_pos_emb carries no
# layer id); resets each new sequence. See _make_prerope_hook.
_PREROPE_CTR = 0
# Lazily-loaded KVQuant calibration cache: {layer_idx: centroids tensor (H*D, nlev)}.
# Populated from KVQ_CACHE on first kvquant use, or injected directly in tests.
_KVQ_CACHE = None


def _group_pad(x, g):
    n = x.shape[2]
    pad = (g - n % g) % g
    if pad:
        x = torch.cat([x, x[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
    return x, n


def _quant_uniform_group(x, bits, g):
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    mn = xg.amin(3, keepdim=True)
    mx = xg.amax(3, keepdim=True)
    qm = 2**bits - 1
    sc = (mx - mn).clamp_min(1e-8) / qm
    xq = ((xg - mn) / sc).round().clamp(0, qm) * sc + mn
    return xq.reshape(B, H, Tg * g, D)[:, :, :n0, :]


def _quant_nf4_group(x, g, nf4):
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    amax = xg.abs().amax(3, keepdim=True).clamp_min(1e-8)
    xn = xg / amax
    lvl = nf4.to(x.device).view(1, 1, 1, 1, 1, -1)
    idx = (xn.unsqueeze(-1) - lvl).abs().argmin(-1)
    deq = nf4.to(x.device)[idx] * amax
    return deq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def _quant_nf4a_group(x, g, nf4):
    """Asymmetric / zero-point NF4: subtract the per-group per-channel MEAN (the DC offset
    that wastes symmetric NF4's codes), NF4-quantize the centered residual, add the mean
    back. Keeps NF4's nonlinear level placement AND handles offset KV distributions -> aims
    to be robust across MHA (Llama) and high-GQA (Qwen) models with one codebook.
    Metadata: per-group mean + absmax (2 fp16 scalars/group/channel vs 1 for symmetric)."""
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    mu = xg.mean(3, keepdim=True)
    xc = xg - mu
    amax = xc.abs().amax(3, keepdim=True).clamp_min(1e-8)
    xn = xc / amax
    lvl = nf4.to(x.device).view(1, 1, 1, 1, 1, -1)
    idx = (xn.unsqueeze(-1) - lvl).abs().argmin(-1)
    deq = nf4.to(x.device)[idx] * amax + mu
    return deq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def _kmeans_1d(x, k, iters, w=None):
    """Per-row 1-D k-means (Lloyd). x: (R, n) -> centroids (R, k). Init = quantiles.

    ``w`` (optional, (R, n)) gives a per-SAMPLE weight: the Lloyd centroid update becomes
    the weighted mean  sum(w*x | assigned) / sum(w | assigned)  instead of the plain mean.
    This is the Fisher-information sensitivity weighting KVQuant uses (high-|gradient|
    activations pull centroids toward them). w=None reproduces the original behaviour
    bit-for-bit. Init quantiles stay unweighted (they only seed the iteration)."""
    qs = torch.linspace(0.0, 1.0, k, device=x.device, dtype=x.dtype)
    cent = torch.quantile(x, qs, dim=1).T.contiguous()  # (R, k)
    for _ in range(iters):
        idx = (x.unsqueeze(-1) - cent.unsqueeze(1)).abs().argmin(-1)  # (R, n)
        oh = torch.nn.functional.one_hot(idx, k).to(x.dtype)  # (R, n, k)
        if w is None:
            cnt = oh.sum(1)  # (R, k)
            s = torch.einsum("rn,rnk->rk", x, oh)  # (R, k)
            new = torch.where(cnt > 0, s / cnt.clamp_min(1), cent)
        else:
            ow = oh * w.unsqueeze(-1)  # (R, n, k) weighted assignment mass
            cnt = ow.sum(1)  # (R, k) total weight per centroid
            s = torch.einsum("rn,rnk->rk", x, ow)  # (R, k) weighted sum
            new = torch.where(cnt > 0, s / cnt.clamp_min(1e-9), cent)
        cent = new
    return cent


def _quant_perchannel_codebook(x, bits, mode):
    """Online per-channel non-uniform codebook fit to this block (quantile or kmeans).

    Fit on a token subsample (codebook is robust to it); assign via searchsorted on the
    sorted codebook -> O(n log nlev), not O(n*nlev).
    """
    B, H, n, D = x.shape
    nlev = 2**bits
    xt = x.permute(0, 1, 3, 2).reshape(-1, n).float()  # (B*H*D, n) per-channel rows
    sub = xt[:, :: max(1, n // 1024)][:, :1024] if n > 1536 else xt
    if mode == "kmeans":
        cent = _kmeans_1d(sub, nlev, KMEANS_ITERS)  # (R, nlev)
    else:  # quantile
        qs = torch.linspace(0.0, 1.0, nlev, device=x.device, dtype=torch.float32)
        cent = torch.quantile(sub, qs, dim=1).T  # (R, nlev)
    cent = cent.sort(dim=1).values.contiguous()
    pos = torch.searchsorted(cent, xt.contiguous()).clamp(1, nlev - 1)
    left = torch.gather(cent, 1, pos - 1)
    right = torch.gather(cent, 1, pos)
    idx = torch.where((xt - left).abs() <= (right - xt).abs(), pos - 1, pos)
    deq = torch.gather(cent, 1, idx)  # (R, n)
    return deq.reshape(B, H, D, n).permute(0, 1, 3, 2).to(x.dtype)


def _dequant_to_centroids(x, cent):
    """Quantize each row of ``x`` (R, n) to its nearest centroid in ``cent`` (R, k) and
    return the dequantized values (R, n). Assignment via searchsorted on the sorted
    codebook -> O(n log k) (same trick as _quant_perchannel_codebook). Idempotent: feeding
    the output back in returns it unchanged (values already sit on centroids)."""
    cs = cent.sort(dim=1).values.contiguous()  # (R, k)
    k = cs.shape[1]
    pos = torch.searchsorted(cs, x.contiguous()).clamp(1, k - 1)  # (R, n)
    left = torch.gather(cs, 1, pos - 1)
    right = torch.gather(cs, 1, pos)
    idx = torch.where((x - left).abs() <= (right - x).abs(), pos - 1, pos)
    return torch.gather(cs, 1, idx)


def _quant_uniform_asym_group(x, bits, g):
    """KIVI key path: per-channel ASYMMETRIC (per-group min/max zero-point) uniform.

    Identical math to the shipped _quant_uniform_group (which is already min/max based, so
    asymmetric) -- kept as a separate, clearly-named entry point so the KIVI baseline owns
    its quantizer and is not coupled to changes in the default uniform path."""
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    mn = xg.amin(3, keepdim=True)
    mx = xg.amax(3, keepdim=True)
    qm = 2**bits - 1
    sc = (mx - mn).clamp_min(1e-8) / qm
    xq = ((xg - mn) / sc).round().clamp(0, qm) * sc + mn
    return xq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def _quant_uniform_sym_group(x, bits, g):
    """SYMMETRIC (abs-max about zero) per-channel uniform -- the symmetric counterpart used
    as the comparison baseline against the asymmetric KIVI path on DC-offset data."""
    B, H, n, D = x.shape
    xp, n0 = _group_pad(x, g)
    Tg = xp.shape[2] // g
    xg = xp.reshape(B, H, Tg, g, D)
    amax = xg.abs().amax(3, keepdim=True).clamp_min(1e-8)
    qm = 2 ** (bits - 1) - 1  # signed symmetric range [-qm, qm]
    sc = amax / qm
    xq = (xg / sc).round().clamp(-qm, qm) * sc
    return xq.reshape(B, H, Tg * g, D)[:, :, :n0, :].to(x.dtype)


def _get_kvq_cache():
    """Lazily load the KVQuant calibration cache (centroids per layer) from KVQ_CACHE.
    Tests may instead inject the module global _KVQ_CACHE directly (skips disk)."""
    global _KVQ_CACHE
    if _KVQ_CACHE is None:
        if not os.path.exists(KVQ_CACHE):
            raise FileNotFoundError(
                f"KVQuant cache {KVQ_CACHE!r} missing; run calibrate_kvquant first "
                f"(or set KVQ_CACHE)."
            )
        _KVQ_CACHE = torch.load(KVQ_CACHE, map_location="cpu")
    return _KVQ_CACHE


def _quant_kvquant_group(x, layer_idx):
    """Eval-time KVQuant key quant: per-channel NEAREST-CENTROID quantization using the
    OFFLINE Fisher-weighted k-means centroids cached for ``layer_idx``. Channels are the
    (head, head_dim) pairs of the stored key cache, i.e. R = H*D rows."""
    cache = _get_kvq_cache()
    cent = cache.get(int(layer_idx))
    if cent is None:
        raise KeyError(
            f"no KVQuant centroids for layer {layer_idx}; calibrate_kvquant must cover "
            f"every layer."
        )
    B, H, n, D = x.shape
    cent = cent.to(device=x.device, dtype=torch.float32)  # (H*D, nlev)
    if cent.shape[0] != H * D:
        raise ValueError(f"KVQuant centroid rows {cent.shape[0]} != H*D {H * D}")
    xt = x.permute(0, 1, 3, 2).reshape(B, H * D, n).float()  # (B, H*D, n) per-channel rows
    out = torch.empty_like(xt)
    for b in range(B):  # benchmark runs B=1; loop keeps memory bounded for B>1
        out[b] = _dequant_to_centroids(xt[b], cent)
    return out.reshape(B, H, D, n).permute(0, 1, 3, 2).to(x.dtype)


def calibrate_kvquant(
    model,
    tokenizer,
    calib_texts,
    n_seq=None,
    max_len=512,
    nlev=None,
    save_path=None,
    iters=None,
):
    """Faithful KVQuant offline calibration (Hooper et al. 2024). GPU/model gated -- there
    is no CUDA or model on the build box, so this is exercised on the cluster, NOT here.

    Per (layer, channel) it collects:
      * the PRE-RoPE key activations (output of self_attn.k_proj, before the rotary
        embedding rotates and smears the per-channel structure NUQ relies on), and
      * a Fisher-information SENSITIVITY weight = mean squared gradient of the LM loss
        w.r.t. that key activation. A full per-activation Hessian is intractable, so we use
        the standard diagonal-Fisher / grad^2 proxy: E[(dL/da)^2]. This requires ONE
        backward pass per calibration sequence (grad enabled; not torch.no_grad).

    It then fits per-channel WEIGHTED k-means centroids (Fisher-weighted Lloyd via the
    extended _kmeans_1d) and writes {layer_idx: centroids (H*D, nlev)} to ``save_path``
    (default KVQ_CACHE) with torch.save, so the eval shard loads them via _get_kvq_cache.

    NOTE (GPU acceptance gate): reproducing the published ~21.06 qasper on Llama-2-7B is
    the validation target for the NRP run and CANNOT be checked locally.
    """
    n_seq = KVQ_CALIB_N if n_seq is None else n_seq
    nlev = (2**KB) if nlev is None else nlev
    iters = KMEANS_ITERS if iters is None else iters
    save_path = KVQ_CACHE if save_path is None else save_path
    device = next(model.parameters()).device

    # Locate the decoder layers and their key projections (Llama/Mistral/Qwen layout).
    layers = model.model.layers
    captured = {}  # layer_idx -> activation tensor (with retained grad)
    handles = []

    def _mk_hook(li):
        def _hook(_module, _inp, out):
            captured[li] = out[0] if isinstance(out, tuple) else out  # (B, n, H_kv*D)
            return out

        return _hook

    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(_mk_hook(li)))

    # Per (layer, channel) running sample + weight buffers (collected on CPU to save VRAM).
    acts = {li: [] for li in range(len(layers))}
    wts = {li: [] for li in range(len(layers))}
    try:
        for text in list(calib_texts)[:n_seq]:
            enc = tokenizer(
                text, truncation=True, max_length=max_len, return_tensors="pt"
            ).to(device)
            # use_cache=False is ESSENTIAL: with the KV cache on, this forward goes
            # through the monkeypatched DynamicCache.update -> qdq_key_block -> the
            # kvquant path, which needs the calibration cache we are creating here
            # (circular -> FileNotFoundError). Calibration only needs the k_proj
            # activations, captured via forward hooks that fire regardless of caching.
            out = model(**enc, labels=enc["input_ids"], use_cache=False)
            # Fisher weight = grad^2 of the loss w.r.t. the key activations. Take grads
            # of the ACTIVATIONS ONLY via autograd.grad (not loss.backward()), so we
            # never materialize gradients for the 7B params (~13GB) -> calibration fits
            # a 24GB GPU instead of needing 48GB (and 3090s are plentiful vs L40S).
            idxs = [li for li in range(len(layers)) if captured.get(li) is not None]
            tensors = [captured[li] for li in idxs]
            grads = torch.autograd.grad(out.loss, tensors, allow_unused=True)
            for li, a, g in zip(idxs, tensors, grads):
                if g is None:
                    continue
                C = a.shape[-1]  # H_kv*D channels
                acts[li].append(a.detach().reshape(-1, C).float().cpu())
                wts[li].append((g.detach().reshape(-1, C).float() ** 2).cpu())
            captured.clear()
            del out, tensors, grads
    finally:
        for h in handles:
            h.remove()

    cache = {}
    for li in range(len(layers)):
        if not acts[li]:
            continue
        A = torch.cat(acts[li], 0)  # (S, C) samples x channels
        W = torch.cat(wts[li], 0)  # (S, C) Fisher weights
        # k-means rows = channels: transpose to (C, S); normalise weights per channel.
        x = A.T.contiguous()  # (C, S)
        w = W.T.contiguous()  # (C, S)
        w = w / w.amax(dim=1, keepdim=True).clamp_min(1e-12)
        cent = _kmeans_1d(x, nlev, iters, w=w)  # (C, nlev)
        cache[li] = cent.sort(dim=1).values.contiguous()
    torch.save(cache, save_path)
    print(f"[kvquant] calibrated {len(cache)} layers -> {save_path}", flush=True)
    return cache


def _calib_texts(n):
    """Calibration corpus for KVQuant: WikiText-2 (the standard KVQuant calib set),
    with a fallback to already-fetched LongBench contexts if the download fails."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 200][:n]
        if len(texts) >= n:
            print(f"[kvquant] calib = {len(texts)} WikiText-2 passages", flush=True)
            return texts
    except Exception as e:
        print(f"[kvquant] wikitext calib failed ({repr(e)[:80]}); using LongBench contexts",
              flush=True)
    texts = []
    for f in sorted(glob.glob(f"{DATADIR}/*.jsonl")):
        for o in load_jsonl(f):
            c = o.get("context") or o.get("input") or ""
            if len(c.strip()) > 200:
                texts.append(c[:4000])
            if len(texts) >= n:
                return texts
    return texts


def qdq_key_block(x):
    """Quantize the full settled key block once (global per-channel outliers + sink)."""
    B, H, n, D = x.shape
    if CODEBOOK == "nf4" and KB == 4:
        hq = _quant_nf4_group(x, G, NF4)
    elif CODEBOOK == "nf4a" and KB == 4:
        hq = _quant_nf4a_group(x, G, NF4)
    elif CODEBOOK in ("quantile", "kmeans"):
        hq = _quant_perchannel_codebook(x, KB, CODEBOOK)
    elif CODEBOOK == "kvquant":
        hq = _quant_kvquant_group(x, _CUR_LAYER)
    elif CODEBOOK == "kivi":
        hq = _quant_uniform_asym_group(x, KB, G)
    else:
        hq = _quant_uniform_group(x, KB, G)
    keep = torch.zeros(B, H, n, D, dtype=torch.bool, device=x.device)
    if SINK > 0:
        keep[:, :, : min(SINK, n), :] = True
    if OUT_FRAC > 0:
        k = max(1, int(round(n * OUT_FRAC)))
        absn = x.abs()
        thr = absn.kthvalue(n - k + 1, dim=2, keepdim=True).values
        keep |= absn >= thr
    return torch.where(keep, x, hq).to(x.dtype)


def qdq_val_block(x):
    mn = x.amin(3, keepdim=True)
    mx = x.amax(3, keepdim=True)
    qm = 2**VB - 1
    sc = (mx - mn).clamp_min(1e-8) / qm
    return (((x - mn) / sc).round().clamp(0, qm) * sc + mn).to(x.dtype)


_orig_update = DynamicCache.update


def _patched_update(self, k, v, li, cache_kwargs=None):
    global _CUR_LAYER
    _CUR_LAYER = li  # thread layer identity to qdq_key_block (KVQuant per-layer codebook)
    fk, fv = _orig_update(self, k, v, li, cache_kwargs)
    if NOQUANT:
        return fk, fv
    # Quantize the settled prefill region ONCE, IN PLACE in the stored cache tensor.
    # `fk` is `self.key_cache[li]`; later decode steps cat new fp16 tokens onto it, so
    # the quantized settled region persists with zero per-step overhead (~fp16 speed).
    if not hasattr(self, "_qdone"):
        self._qdone = set()
    if li not in self._qdone:
        T = fk.shape[2]
        n = max(0, T - HOT)
        if n > 0:
            if not PREROPE:  # post-RoPE keys quantized here; pre-RoPE done in the rope hook
                fk[:, :, :n, :] = qdq_key_block(fk[:, :, :n, :])
            fv[:, :, :n, :] = qdq_val_block(fv[:, :, :n, :])
        self._qdone.add(li)
    return fk, fv


DynamicCache.update = _patched_update


def _make_prerope_hook(orig):
    def _hook(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        global _CUR_LAYER, _PREROPE_CTR
        if not NOQUANT:
            T = k.shape[2]
            if T > HOT:  # prefill: quantize settled keys once, pre-RoPE
                # The shared apply_rotary_pos_emb carries no layer id, but during the single
                # prefill forward it fires once per layer in order 0..L-1; count those calls
                # so kvquant looks up the right per-layer codebook. The counter resets on the
                # next decode call (T<=HOT) below, i.e. at the start of each new sequence.
                _CUR_LAYER = _PREROPE_CTR
                _PREROPE_CTR += 1
                n = T - HOT
                k = k.clone()
                k[:, :, :n, :] = qdq_key_block(k[:, :, :n, :])
            else:
                _PREROPE_CTR = 0
        return orig(q, k, cos, sin, position_ids, unsqueeze_dim)

    return _hook


def _install_prerope():
    """Quantize keys in **pre-RoPE** space: hook ``apply_rotary_pos_emb`` so the settled
    prefill keys are quantized before the position rotation is applied (RoPE smears the
    per-channel structure that per-channel quantization relies on).

    HF defines ``apply_rotary_pos_emb`` per model family and the attention forward looks
    it up from the module globals at call time, so monkeypatching the module attribute is
    sufficient. Patch every family we benchmark that is importable in this transformers."""
    mods = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
    ]
    patched = []
    import importlib

    for mp in mods:
        try:
            m = importlib.import_module(mp)
        except Exception:
            continue
        if hasattr(m, "apply_rotary_pos_emb"):
            m.apply_rotary_pos_emb = _make_prerope_hook(m.apply_rotary_pos_emb)
            patched.append(mp.split(".")[-2])
    print(f"[prerope] patched: {patched}", flush=True)


if PREROPE:
    _install_prerope()


def load_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def build_chat(tok, prompt):
    if not CHAT:
        return prompt
    if "llama2" in MODEL_KEY:
        return f"[INST]{prompt}[/INST]"
    try:  # Mistral / Qwen / others: use the tokenizer chat template
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt


def main():
    print(f"[shard {SHARD}/{NSH}] TAG={TAG} MODEL={MODEL_KEY} KB={KB} HOT={HOT} "
          f"SINK={SINK} OUT={OUT_FRAC} NUQ={NUQ} NOQUANT={NOQUANT}", flush=True)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, config=config, torch_dtype=torch.float16,
        attn_implementation="sdpa", low_cpu_mem_usage=True, trust_remote_code=True,
    ).cuda().eval()
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
    except Exception as e:  # some fast tokenizers need a newer tokenizers lib; fall back
        print(f"[tok] fast load failed ({repr(e)[:80]}); using slow tokenizer", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)
    device = torch.device("cuda:0")
    d2p = json.load(open(f"{LBROOT}/config/dataset2prompt.json"))
    d2m = json.load(open(f"{LBROOT}/config/dataset2maxlen.json"))
    no_chat = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
    # KVQuant: run the offline Fisher calibration ONCE before eval if the cache is
    # missing (the eval path needs per-layer centroids). Self-contained -- no separate
    # driver. Single-shard runs only; multi-shard would race on the cache file.
    if CODEBOOK == "kvquant" and _KVQ_CACHE is None and not os.path.exists(KVQ_CACHE):
        print(f"[kvquant] calibrating (n_seq={KVQ_CALIB_N}) -> {KVQ_CACHE}", flush=True)
        calibrate_kvquant(model, tok, _calib_texts(KVQ_CALIB_N))
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    for dataset in DATASETS:
        data = load_jsonl(f"{DATADIR}/{dataset}.jsonl")
        pf = d2p[dataset]
        mg = int(d2m[dataset])
        fo = open(f"{OUT}/{dataset}.{SHARD}.jsonl", "w")
        for gi, o in enumerate(data):
            if gi % NSH != SHARD:
                continue
            if _MAXGEN > 0:
                mg = _MAXGEN  # override the LongBench per-task max_new_tokens (ablation)
            prompt = pf.format(**o)
            tp = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tp) > MAXLEN:
                h = MAXLEN // 2
                prompt = tok.decode(tp[:h], skip_special_tokens=True) + tok.decode(
                    tp[-h:], skip_special_tokens=True
                )
            if dataset not in no_chat:
                prompt = build_chat(tok, prompt)
            inp = tok(prompt, truncation=False, return_tensors="pt").to(device)
            cl = inp.input_ids.shape[-1]
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=mg, num_beams=1, do_sample=False,
                    temperature=1.0, pad_token_id=tok.eos_token_id,
                )[0]
            pred = tok.decode(out[cl:], skip_special_tokens=True)
            fo.write(json.dumps({"idx": gi, "pred": pred, "answers": o["answers"],
                                 "all_classes": o["all_classes"]}) + "\n")
            fo.flush()
        fo.close()
        print(f"[shard {SHARD}] {dataset} done", flush=True)
    print(f"SHARD_{SHARD}_DONE", flush=True)


if __name__ == "__main__":
    main()
