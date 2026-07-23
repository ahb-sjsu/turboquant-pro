# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""The KV identity profile (2.0 roadmap P1-M1) — the gate for persistence.

A persisted KV block is only reusable when **every** configuration that can
change its contents matches. This module defines the canonical,
content-addressed identity for a KV cache tier:

* :class:`KVIdentityProfile` — the configuration layer: model repo/revision,
  weight and tokenizer fingerprints, architecture, adapters, RoPE, attention
  backend and KV-layout version, parallelism, dtype, block size, head config,
  sliding window, quantization discipline, encoder version.
* :func:`prefix_block_hashes` — the content layer: token IDs (never source
  text) hashed as a per-block chain seeded by the profile digest, so two
  requests sharing a token prefix share leading hashes (block-granular prefix
  matching) and *nothing* matches across different configurations.

**Governing rule, enforced structurally: uncertain compatibility ⇒ cache miss
and recomputation — never best-effort decode.** A profile with any unknown
(``None``) field is *incomplete*; an incomplete profile is compatible with
nothing — including itself. There is no "close enough" comparison anywhere in
this module by design: compatibility is digest equality between two complete
profiles, full stop.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "KVIdentityProfile",
    "IncompatibleProfile",
    "prefix_block_hashes",
]

_SCHEMA = "tqp-kv-identity/1"  # bump on any field addition/removal/renaming


class IncompatibleProfile(Exception):
    """Raised where a silent miss is not possible (e.g. explicit import)."""


def _canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace drift, NaN/Inf rejected.

    ``allow_nan=False`` makes non-finite floats a hard error rather than a
    silently platform-dependent token — canonical means canonical.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


@dataclass(frozen=True)
class KVIdentityProfile:
    """The configuration layer of the KV cache key.

    Every field participates in the digest. ``None`` means *unknown*, and
    unknown is contagious: :attr:`is_complete` is False and
    :meth:`compatible` returns False against everything. Extractors that
    cannot determine a field MUST leave it ``None`` — never guess a default.

    Dict-valued fields (``rope``, ``quant``) are canonicalized recursively;
    put every parameter that changes bytes in there (scaling factors,
    per-channel settings, outlier fractions, …).
    """

    # --- model state -----------------------------------------------------
    model_repo: str | None = None
    model_revision: str | None = None
    weight_fingerprint: str | None = None
    tokenizer_fingerprint: str | None = None
    architecture: str | None = None
    adapter_identity: str | None = None  # LoRA/adapter id+hash; "" = none
    # --- positional / attention configuration ---------------------------
    rope: dict | None = None  # base, scaling type + factors, max pos
    attention_backend: str | None = None
    kv_layout_version: str | None = None
    sliding_window: int | None = None  # 0 = disabled
    # --- shape / placement ----------------------------------------------
    n_layers: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None  # GQA/MQA; == n_heads for MHA
    head_dim: int | None = None
    tp_size: int | None = None
    pp_size: int | None = None
    kv_dtype: str | None = None
    block_size: int | None = None
    # --- turboquant side --------------------------------------------------
    quant: dict | None = None  # key/value plugin names + full parameters
    encoder_version: str | None = None  # turboquant-pro version string
    schema: str = field(default=_SCHEMA)

    # ---------------------------------------------------------------- api
    @property
    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(k for k, v in asdict(self).items() if v is None and k != "schema")

    @property
    def is_complete(self) -> bool:
        return not self.unknown_fields

    def digest(self) -> str:
        """Content-addressed sha256 over the canonical encoding.

        Defined for incomplete profiles too (unknowns serialize as null) so
        diagnostics can name what a store *claims* to be — but compatibility
        never consults the digest of an incomplete profile.
        """
        return hashlib.sha256(_canonical(asdict(self)).encode()).hexdigest()

    def compatible(self, other: KVIdentityProfile | None) -> bool:
        """Digest equality between two COMPLETE profiles; anything else False.

        The deliberate asymmetry with ``==``: an incomplete profile equals
        itself as a dataclass but is compatible with nothing — uncertain ⇒
        miss.
        """
        if other is None or not self.is_complete or not other.is_complete:
            return False
        return self.digest() == other.digest()

    def require_compatible(self, other: KVIdentityProfile | None) -> None:
        """Raise :class:`IncompatibleProfile` (with the reason) on mismatch."""
        if self.compatible(other):
            return
        if other is None:
            raise IncompatibleProfile("no profile supplied")
        missing = self.unknown_fields + other.unknown_fields
        if missing:
            raise IncompatibleProfile(
                f"incomplete profile: unknown fields {sorted(set(missing))} — "
                "uncertain compatibility is a miss, never a best-effort decode"
            )
        diff = [k for k, v in asdict(self).items() if asdict(other).get(k) != v]
        raise IncompatibleProfile(f"profile mismatch in fields {diff}")

    # -------------------------------------------------------- extractors
    @classmethod
    def from_vllm_config(
        cls, vllm_config: Any, *, quant: dict | None = None
    ) -> KVIdentityProfile:
        """Best-effort extraction from a vLLM config object.

        Anything not confidently determined stays ``None`` — which makes the
        profile incomplete and therefore unusable for reuse until the deployer
        supplies the missing facts (e.g. weight/tokenizer fingerprints, which
        vLLM does not expose). Guessing here would convert "uncertain" into
        "wrong-prefix reuse", the one unacceptable failure.
        """
        from turboquant_pro import __version__

        model = getattr(vllm_config, "model_config", None)
        parallel = getattr(vllm_config, "parallel_config", None)
        cache = getattr(vllm_config, "cache_config", None)

        def g(obj, *names):
            for n in names:
                v = getattr(obj, n, None)
                if v is not None:
                    return v
            return None

        hf = g(model, "hf_config")
        rope = None
        if hf is not None:
            rope = {
                "theta": g(hf, "rope_theta"),
                "scaling": g(hf, "rope_scaling"),
                "max_position_embeddings": g(hf, "max_position_embeddings"),
            }
        sliding = g(hf, "sliding_window") if hf is not None else None
        return cls(
            model_repo=g(model, "model"),
            model_revision=g(model, "revision"),
            weight_fingerprint=None,  # not derivable from config alone
            tokenizer_fingerprint=None,  # ditto
            architecture=(
                (g(hf, "architectures") or [None])[0] if hf is not None else None
            ),
            adapter_identity="",  # base model; LoRA integration sets this
            rope=rope,
            attention_backend=None,  # worker-side runtime fact
            kv_layout_version=None,
            sliding_window=0 if sliding is None and hf is not None else sliding,
            n_layers=g(hf, "num_hidden_layers") if hf is not None else None,
            n_heads=g(hf, "num_attention_heads") if hf is not None else None,
            n_kv_heads=(
                g(hf, "num_key_value_heads", "num_attention_heads")
                if hf is not None
                else None
            ),
            head_dim=g(hf, "head_dim") if hf is not None else None,
            tp_size=g(parallel, "tensor_parallel_size"),
            pp_size=g(parallel, "pipeline_parallel_size"),
            kv_dtype=str(g(cache, "cache_dtype")) if cache is not None else None,
            block_size=g(cache, "block_size"),
            quant=quant,
            encoder_version=__version__,
        )


def prefix_block_hashes(
    profile: KVIdentityProfile,
    token_ids: Sequence[int],
    *,
    block_size: int = 16,
) -> list[str]:
    """Chained per-block content hashes of a token prefix under a profile.

    ``h_0 = H(profile.digest())``; ``h_i = H(h_{i-1} || tokens of block i)``.
    Two requests sharing a token prefix (same profile) share leading hashes —
    block-granular prefix matching, the vLLM prefix-cache shape — and any
    profile difference changes *every* hash. Only whole blocks are hashed:
    a partial trailing block is not addressable (it will re-hash when full).

    Raises :class:`IncompatibleProfile` for incomplete profiles: content
    addresses must never be minted under uncertain identity.
    """
    if not profile.is_complete:
        raise IncompatibleProfile(
            f"cannot mint cache keys under an incomplete profile "
            f"(unknown: {sorted(profile.unknown_fields)})"
        )
    ids = [int(t) for t in token_ids]
    out: list[str] = []
    h = hashlib.sha256(profile.digest().encode()).hexdigest()
    for s in range(0, len(ids) - len(ids) % block_size, block_size):
        block = ids[s : s + block_size]
        h = hashlib.sha256((h + ":" + _canonical(block)).encode()).hexdigest()
        out.append(h)
    return out
