# TurboQuant Pro: STRATA Phase 2 — per-area remedies (mechanism selects remedy)
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""STRATA Phase 2 (docs/STRATA_RFC.md §3): the anatomy is a prescription pad.

Two remedies, matched to the two hub mechanisms:

- **CSLS / mutual proximity** (density-mechanism hubs): one precomputed
  scalar per record — ``r_k(x)``, the mean similarity of ``x`` to its k
  nearest neighbours — subtracted at scoring time. Carried on disk as the
  TQE1 optional trailer ``0x10`` (``hubness-scalar/1``, RFC §9a: ID and
  syntax frozen, semantics provisional until the efficacy notebook).
- **Per-area (localized) centering** (centrality-mechanism hubs): one
  stored centroid per area. Centroids are ENCODER PARAMETERS: they enter
  the identity profile (``area_codec_params``), changing them is an epoch
  event, and — per the §3 threat clause — they are attacker-relevant
  state, never write-path trust.

Approved claim shape: "Δ anti-hub recall measured in <artifact> on
<corpus>". Banned: any unconditional improvement claim.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

# Registered estimators for the r_k scalar (u8 in the trailer payload).
RK_ESTIMATORS = {"exact_knn": 0, "adc_candidates": 1}

_TRAILER_ID = 0x10
_TRAILER_NAME = "hubness-scalar/1"


def mutual_proximity_scalar(
    base: np.ndarray, k: int = 10, *, estimator: str = "exact_knn"
) -> np.ndarray:
    """r_k(x): mean cosine similarity of each row to its k nearest neighbours.

    The CSLS local-scale scalar (Conneau et al. 2018). One float32 per
    record; the estimator is declared and travels in the trailer payload.
    """
    from .anatomy import knn_exact

    if estimator not in RK_ESTIMATORS:
        raise KeyError(f"unregistered r_k estimator {estimator!r}")
    x = np.ascontiguousarray(base, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    d, _ = knn_exact(x, x, k, exclude_self=True)
    # Unit vectors: cos = 1 - d^2/2.
    return (1.0 - 0.5 * d.astype(np.float64) ** 2).mean(axis=1).astype(np.float32)


def csls_rescore(
    scores: np.ndarray, candidate_rk: np.ndarray, *, weight: float = 1.0
) -> np.ndarray:
    """CSLS correction on similarity scores: score' = score − w·r_k(candidate).

    The query-side r term of full CSLS is constant per query and cannot
    change that query's ranking; only the candidate-side scalar is applied.
    """
    return scores - weight * candidate_rk


def pack_hubness_trailer(
    r_k: np.ndarray, k: int, *, estimator: str = "exact_knn"
) -> bytes:
    """Serialize ``hubness-scalar/1`` as a length-delimited optional trailer.

    Layout: u8 trailer_id (0x10) · u32 payload_len · payload · sha256/8 of
    (id || payload). Payload: u8 k · u8 estimator_id · u32 n · n×f32 r_k.
    Unknown-optional ⇒ skip: readers that don't know 0x10 stay conformant.
    """
    r = np.ascontiguousarray(r_k, dtype=np.float32)
    payload = struct.pack("<BBI", k, RK_ESTIMATORS[estimator], len(r)) + r.tobytes()
    head = struct.pack("<BI", _TRAILER_ID, len(payload))
    digest = hashlib.sha256(bytes([_TRAILER_ID]) + payload).digest()[:8]
    return head + payload + digest


def unpack_hubness_trailer(blob: bytes) -> dict:
    """Parse a ``hubness-scalar/1`` trailer; integrity failure ⇒ ValueError."""
    tid, plen = struct.unpack_from("<BI", blob, 0)
    if tid != _TRAILER_ID:
        raise ValueError(f"not a {_TRAILER_NAME} trailer (id {tid:#x})")
    payload = blob[5 : 5 + plen]
    digest = blob[5 + plen : 5 + plen + 8]
    if hashlib.sha256(bytes([tid]) + payload).digest()[:8] != digest:
        raise ValueError(f"{_TRAILER_NAME} trailer failed its own hash")
    k, est, n = struct.unpack_from("<BBI", payload, 0)
    r = np.frombuffer(payload, dtype=np.float32, count=n, offset=6)
    est_name = {v: kk for kk, v in RK_ESTIMATORS.items()}[est]
    return {"k": int(k), "estimator": est_name, "r_k": r.copy()}


class AreaCentering:
    """Per-area localized centering: subtract the area's stored centroid.

    The v1.4.0 centering theorem at region scale — a local DC offset a
    global codebook wastes codes on. Centroids are encoder parameters:
    ``params_digest`` goes into the identity profile; a reader/searcher
    holding a different digest MUST refuse (epoch event, not a tweak).
    """

    def __init__(self, centroids: dict[str, np.ndarray], area_map_digest: str):
        self.centroids = {
            a: np.asarray(c, dtype=np.float32) for a, c in centroids.items()
        }
        self.area_map_digest = area_map_digest

    @classmethod
    def fit(cls, base: np.ndarray, area_map) -> AreaCentering:
        x = np.ascontiguousarray(base, dtype=np.float32)
        lab = np.asarray(area_map.labels)
        cents = {a: x[lab == a].mean(0) for a in area_map.areas}
        return cls(cents, area_map.digest)

    @property
    def params_digest(self) -> str:
        h = hashlib.sha256(self.area_map_digest.encode())
        for a in sorted(self.centroids):
            h.update(a.encode())
            h.update(np.ascontiguousarray(self.centroids[a]).tobytes())
        return h.hexdigest()

    def apply(self, x: np.ndarray, areas: list[str]) -> np.ndarray:
        """Center rows by their own area's centroid (renormalized)."""
        out = np.ascontiguousarray(x, dtype=np.float32).copy()
        lab = np.asarray(areas)
        for a in np.unique(lab):
            if a not in self.centroids:
                raise KeyError(f"no centroid for area {a!r} under this map")
            out[lab == a] -= self.centroids[a]
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)
