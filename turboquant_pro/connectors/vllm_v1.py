# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""``TurboQuantKVConnector`` — vLLM V1 KV-connector integration (2.0 Pillar 1).

vLLM's V1 engine exposes a first-class connector interface
(``KVConnectorBase_V1``) with two roles: a **scheduler-side** instance that
decides how many externally-held tokens a request can reuse, and a
**worker-side** instance that saves finished KV layers out of, and loads
matched prefixes back into, the paged KV cache. Registering a connector makes
it available to any deployment by configuration alone::

    from turboquant_pro.connectors import register
    register()   # once, e.g. in a vLLM plugin or sitecustomize

    # then:
    #   vllm serve <model> --kv-transfer-config \
    #     '{"kv_connector": "TurboQuantConnector", "kv_role": "kv_both"}'

**2.0 MVP semantics — the offload/persistence tier.** ``save_kv_layer``
quantizes each layer's KV block tensor with the (A2)-correct disciplines
(**per-channel keys**, **polar values** — the architecture this project's KV
findings mandate; see ``docs/KV_KEYS_FINDING.md``) into a host-side
:class:`TurboQuantBlockStore`; ``start_load_kv`` restores previously saved
prefixes on a scheduler match. ~4-5x more cached prefix per host byte, and the
store speaks TQE1 records so a saved tier is persistable and shippable
(roadmap Pillar 2, ``kv_block`` profile).

Coherence rule, unchanged here: enabling the connector for a model should be
gated on the **(A2) probe / behavioral agreement**, never on reconstruction
cosine — ``tqp probe`` / ``tqp plan kv`` emit the verdict; roadmap M4 wires
that into an emitted ``--kv-transfer-config``.

The module imports (and subclasses) vLLM lazily: without vLLM installed it
degrades to an importable shim with the identical protocol surface, which is
what the in-tree tests exercise. The real-engine end-to-end lane runs in CI
against pinned vLLM versions (roadmap P1); vLLM's connector API is marked
experimental upstream, so signatures here accept liberal ``*args/**kwargs``
and are re-validated per pin.

**Safety scope of this scaffold (read before extending).** The store is
in-process and keyed by request id: blocks never survive the process, never
cross nodes, and are evicted on request finish — so no wrong-prefix reuse is
possible *yet*. Any persistence or cross-node feature MUST first implement
the **KV identity profile** (roadmap P1-M1: content-addressed key over model
revision + weight/tokenizer fingerprints + token IDs + adapter/RoPE/layout/
parallelism/dtype/discipline/encoder-version), under the governing rule
*uncertain compatibility ⇒ cache miss and recomputation* — never best-effort
decode. Failure semantics (P1-M2) likewise gate any beta: corruption ⇒ miss,
timeout ⇒ recompute, connector failure never fails the request.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CONNECTOR_NAME = "TurboQuantConnector"


# --------------------------------------------------------------------- store
@dataclass
class _BlockRecord:
    """One saved KV layer block: compressed payloads + enough to restore."""

    key_payload: Any
    value_payload: Any
    shape: tuple[int, ...]
    dtype: str
    num_tokens: int
    head_dim: int
    n_heads: int


@dataclass
class TurboQuantBlockStore:
    """Host-side store of quantized KV blocks, keyed by (request, layer).

    Engine-agnostic on purpose: the vLLM connector and the future SGLang
    adapter mount the same store. Keys are compressed with the ``per_channel``
    plugin and values with ``polar`` — the disciplines the (A2) probe selects
    for attention KV — via the public plugin registry, so conformance-kit
    guarantees apply to what production traffic runs through.

    Identity (roadmap P1-M1): the store optionally carries a
    :class:`~turboquant_pro.connectors.identity.KVIdentityProfile`. In-process
    operation does not require one (nothing outlives the process), but the
    persistence hooks — :meth:`export_state` / :meth:`import_state` — REQUIRE
    a complete, matching profile on both sides: uncertain ⇒ the import is
    refused, i.e. a miss, never a best-effort decode.
    """

    key_plugin: str = "per_channel"
    value_plugin: str = "polar"
    quantizer_config: dict = field(default_factory=dict)
    profile: Any = None  # KVIdentityProfile | None

    def __post_init__(self) -> None:
        self._records: dict[tuple[str, str], _BlockRecord] = {}
        self._tokens: dict[str, int] = {}
        self._lock = threading.Lock()
        self._q_cache: dict[str, Any] = {}

    # -- internals ---------------------------------------------------------
    def _quantizer(self, name: str, head_dim: int, n_heads: int):
        """Per-geometry quantizer: KV plugins size rotations/grids to
        ``(head_dim, n_heads)`` (same convention as the conformance kit)."""
        key = (name, head_dim, n_heads)
        q = self._q_cache.get(key)
        if q is None:
            from turboquant_pro import create_quantizer

            cfg = dict(self.quantizer_config)
            cfg.setdefault("head_dim", head_dim)
            cfg.setdefault("n_heads", n_heads)
            try:
                q = create_quantizer(name, **cfg)
            except TypeError:  # plugin doesn't take geometry/config kwargs
                q = create_quantizer(name)
            self._q_cache[key] = q
        return q

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):  # torch tensor, any device
            x = x.detach().to("cpu").float().numpy()
        return np.ascontiguousarray(x, dtype=np.float32)

    # -- API ---------------------------------------------------------------
    @staticmethod
    def _as_bhsd(x: np.ndarray) -> np.ndarray:
        """Normalize any engine layout to the plugins' (B, H, S, D) contract.

        3-D (H, S, D) gains a singleton batch; >4-D layouts fold their leading
        dims into the batch axis. Tokens are axis -2, head_dim axis -1 — the
        convention every current vLLM attention backend uses per layer.
        """
        if x.ndim == 4:
            return x
        if x.ndim < 4:
            return x.reshape((1,) * (4 - x.ndim) + x.shape)
        return x.reshape((-1,) + x.shape[-3:])

    def save(self, request_id: str, layer_name: str, keys: Any, values: Any) -> None:
        """Quantize and store one layer's KV for ``request_id``."""
        k = self._to_numpy(keys)
        v = self._to_numpy(values)
        k4, v4 = self._as_bhsd(k), self._as_bhsd(v)
        d, h = int(k4.shape[-1]), int(k4.shape[1])
        rec = _BlockRecord(
            key_payload=self._quantizer(self.key_plugin, d, h).compress(k4),
            value_payload=self._quantizer(self.value_plugin, d, h).compress(v4),
            shape=tuple(k.shape),
            dtype="float32",
            num_tokens=int(k4.shape[-2]),
            head_dim=d,
            n_heads=h,
        )
        with self._lock:
            self._records[(request_id, layer_name)] = rec
            self._tokens[request_id] = rec.num_tokens

    def load(
        self, request_id: str, layer_name: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Dequantize one layer's KV, or ``None`` when nothing is stored.

        Failure semantics (roadmap P1-M2): any decode error is a MISS — the
        record is dropped, ``None`` is returned, and the caller recomputes.
        A load must never propagate an exception into the serving path.
        """
        with self._lock:
            rec = self._records.get((request_id, layer_name))
        if rec is None:
            return None
        try:
            k = self._quantizer(self.key_plugin, rec.head_dim, rec.n_heads).decompress(
                rec.key_payload
            )
            v = self._quantizer(
                self.value_plugin, rec.head_dim, rec.n_heads
            ).decompress(rec.value_payload)
            return (
                np.asarray(k, dtype=np.float32).reshape(rec.shape),
                np.asarray(v, dtype=np.float32).reshape(rec.shape),
            )
        except Exception:
            logger.warning(
                "corrupt/undecodable KV record (%s, %s): treating as miss",
                request_id,
                layer_name,
                exc_info=True,
            )
            with self._lock:
                self._records.pop((request_id, layer_name), None)
            return None

    # -- persistence hooks (P1-M1 gated) -----------------------------------
    def export_state(self) -> dict:
        """Snapshot for handoff to another store — REQUIRES a complete profile.

        Payloads are pickled with a per-record sha256; this is an in-process /
        same-trust-domain handoff primitive. Cross-trust persistence is the
        TQE1 ``kv_block`` record profile (roadmap P2), not pickle.
        """
        import pickle

        from .identity import IncompatibleProfile

        if self.profile is None or not self.profile.is_complete:
            raise IncompatibleProfile(
                "export requires a complete KVIdentityProfile "
                "(uncertain identity must not be persisted)"
            )
        with self._lock:
            records = {}
            for key, rec in self._records.items():
                blob = pickle.dumps(rec)
                records["\x00".join(key)] = {
                    "blob": blob,
                    "sha256": __import__("hashlib").sha256(blob).hexdigest(),
                }
            return {
                "schema": "tqp-kv-store-state/1",
                "profile_digest": self.profile.digest(),
                "records": records,
                "tokens": dict(self._tokens),
            }

    def import_state(self, state: dict) -> int:
        """Adopt exported records; returns how many were accepted.

        Gate order: profile compatibility first (mismatch/incomplete raises
        :class:`IncompatibleProfile` — the caller treats it as a total miss),
        then per-record integrity (a corrupt record is skipped — a miss —
        while intact records still load).
        """
        import hashlib as _hashlib
        import pickle

        from .identity import IncompatibleProfile

        if self.profile is None or not self.profile.is_complete:
            raise IncompatibleProfile("import requires a complete profile")
        if state.get("profile_digest") != self.profile.digest():
            raise IncompatibleProfile(
                "profile digest mismatch — refusing every record (safe miss)"
            )
        accepted = 0
        for key, item in state.get("records", {}).items():
            blob = item["blob"]
            if _hashlib.sha256(blob).hexdigest() != item["sha256"]:
                logger.warning("integrity failure on %r: skipped (miss)", key)
                continue
            rec = pickle.loads(blob)
            rid, layer = key.split("\x00", 1)
            with self._lock:
                self._records[(rid, layer)] = rec
                self._tokens[rid] = rec.num_tokens
            accepted += 1
        return accepted

    def matched_tokens(self, request_id: str) -> int:
        """Tokens this store can restore for ``request_id`` (0 if unknown)."""
        with self._lock:
            return self._tokens.get(request_id, 0)

    def evict(self, request_id: str) -> int:
        """Drop every layer stored for ``request_id``; returns records freed."""
        with self._lock:
            gone = [kk for kk in self._records if kk[0] == request_id]
            for kk in gone:
                del self._records[kk]
            self._tokens.pop(request_id, None)
        return len(gone)

    def stats(self) -> dict:
        with self._lock:
            return {
                "requests": len(self._tokens),
                "records": len(self._records),
                "key_plugin": self.key_plugin,
                "value_plugin": self.value_plugin,
            }


# ------------------------------------------------------------------ vLLM glue
def _resolve_base() -> tuple[type, Any]:
    """The installed vLLM's connector base + role enum, or the local shim."""
    try:  # pragma: no cover - exercised only with vLLM installed
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
            KVConnectorRole,
        )

        return KVConnectorBase_V1, KVConnectorRole
    except Exception:

        class _ShimRole:
            SCHEDULER = "scheduler"
            WORKER = "worker"

        class _ShimBase:  # the protocol surface, importable without vLLM
            def __init__(self, vllm_config: Any = None, role: Any = None):
                self.vllm_config = vllm_config
                self.role = role

        return _ShimBase, _ShimRole


_Base, KVConnectorRole = _resolve_base()


class TurboQuantKVConnector(_Base):  # type: ignore[misc, valid-type]
    """vLLM V1 connector backed by :class:`TurboQuantBlockStore`.

    One class serves both roles (vLLM instantiates it twice); role-specific
    methods are no-ops for the other role. The 2.0 MVP is synchronous —
    ``wait_*`` return immediately — with async spill/prefetch queued behind
    the fused-dequant milestone (roadmap M3).
    """

    def __init__(self, vllm_config: Any = None, role: Any = None, **kwargs: Any):
        try:
            super().__init__(vllm_config, role)
        except TypeError:  # older/newer base signature drift
            super().__init__()
            self.vllm_config = vllm_config
            self.role = role
        cfg = {}
        extra = getattr(
            getattr(vllm_config, "kv_transfer_config", None),
            "kv_connector_extra_config",
            None,
        )
        if isinstance(extra, dict):
            cfg = dict(extra)
        self.store = kwargs.get("store") or TurboQuantBlockStore(
            key_plugin=cfg.get("key_plugin", "per_channel"),
            value_plugin=cfg.get("value_plugin", "polar"),
            quantizer_config=cfg.get("quantizer_config", {}),
        )
        logger.info(
            "TurboQuantKVConnector initialised (role=%s, store=%s)",
            role,
            self.store.stats(),
        )

    # ---- scheduler-side protocol ----------------------------------------
    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int, *args: Any, **kwargs: Any
    ) -> tuple[int, bool]:
        """Tokens the store can supply beyond vLLM's own prefix cache."""
        rid = getattr(request, "request_id", None) or str(request)
        matched = self.store.matched_tokens(rid)
        extra = max(0, matched - int(num_computed_tokens))
        return extra, False  # synchronous load: nothing pending async

    def update_state_after_alloc(self, *args: Any, **kwargs: Any) -> None:
        return None

    def build_connector_meta(self, scheduler_output: Any = None, **kwargs: Any):
        """Opaque metadata for the worker; MVP ships request ids to load."""
        return {"version": 1}

    def request_finished(
        self, request: Any = None, block_ids: Any = None, **kwargs: Any
    ) -> tuple[bool, dict | None]:
        """Blocks are not held after finish; the store owns its own copy."""
        return False, None

    # ---- worker-side protocol -------------------------------------------
    def start_load_kv(self, forward_context: Any = None, **kwargs: Any) -> None:
        """Restore saved prefixes into the paged cache (synchronous MVP)."""
        return None

    def wait_for_layer_load(self, layer_name: str, **kwargs: Any) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        """Quantize one layer's (keys, values) pair into the store.

        vLLM hands the layer's paged KV tensor; the MVP treats index 0 as keys
        and index 1 as values (the layout every current attention backend
        uses) and stores under the request id carried in ``attn_metadata``.
        """
        rid = str(getattr(attn_metadata, "request_id", kwargs.get("request_id", "_")))
        keys, values = kv_layer[0], kv_layer[1]
        self.store.save(rid, layer_name, keys, values)

    def wait_for_save(self, **kwargs: Any) -> None:
        return None

    def get_finished(
        self, finished_req_ids: Any = None, **kwargs: Any
    ) -> tuple[set | None, set | None]:
        if finished_req_ids:
            for rid in finished_req_ids:
                self.store.evict(str(rid))
        return None, None


def register(name: str = _CONNECTOR_NAME) -> bool:
    """Register with vLLM's ``KVConnectorFactory`` (idempotent).

    Returns ``True`` when vLLM is present and registration succeeded, so
    callers can log-and-continue in environments without vLLM.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )
    except Exception:
        logger.info("vLLM not installed; TurboQuantConnector not registered")
        return False
    try:
        KVConnectorFactory.register_connector(
            name, "turboquant_pro.connectors.vllm_v1", "TurboQuantKVConnector"
        )
    except ValueError:  # already registered — idempotent by intent
        pass
    return True
