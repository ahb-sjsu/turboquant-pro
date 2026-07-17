# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Production vector-index lifecycle for Track 1 (``tqp index``).

A benchmark index lives in RAM and is thrown away. A production RAG index must
persist, grow, forget, stay honest about storage, survive format upgrades, and
prove it still ranks well. :class:`TQEIndex` is that index: a compressed-domain
ADC search structure (PCA-Matryoshka + TurboQuant, the Track-1 recipe) with a
full lifecycle over the versioned :mod:`turboquant_pro.index_file` container.

    ingest -> search -> update -> compact -> migrate -> certify -> monitor

- **create / open** — fit once, persist the PCA basis + codes; reopen with no refit.
- **add** — append new vectors, compressed by the *same* basis (no re-fit, so ids
  and scores stay comparable).
- **delete** — tombstone by external id (O(1)); storage is reclaimed at compaction.
- **compact** — physically drop tombstoned rows, so bytes-on-disk stay honest.
- **migrate** — upgrade the on-disk format (v1 positional ids -> v2 explicit ids +
  tombstones), exercising the versioning under a real upgrade.
- **certify** — reconstruct a sample and emit a distribution-free rank certificate
  (the metric retrieval consumes), never reconstruction cosine.
- **drift** — flag a stale PCA basis before it silently costs recall.

Acceptance everywhere is rank fidelity (recall / the rank certificate), never
reconstruction cosine — see ``docs/KV_KEYS_FINDING.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .adc_index import ADCIndex
from .index_file import read_container, read_directory, write_container
from .pca import PCAMatryoshka
from .rank_certificate import RankCertificate, certificate_from_embeddings

CURRENT_VERSION = 2  # v1: implicit positional ids; v2: explicit ids + tombstones
_SUPPORTED = (1, 2)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arr_bytes(a: np.ndarray) -> tuple[bytes, dict]:
    a = np.ascontiguousarray(a)
    return a.tobytes(), {"dtype": a.dtype.str, "shape": list(a.shape)}


def _arr_from(blob: bytes, spec: dict) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])


@dataclass
class DriftReport:
    """Whether the stored PCA basis is still a good fit for new data."""

    mean_shift: float  # ||mean(new) - stored_mean|| / ||stored_mean||
    retained_var_fit: float  # variance fraction the basis kept at build time
    retained_var_new: float  # variance fraction it keeps on the new batch
    retained_var_drop: float  # fit - new (positive = basis is losing signal)
    stale: bool  # drop exceeds the threshold

    def as_dict(self) -> dict:
        return {
            "mean_shift": self.mean_shift,
            "retained_var_fit": self.retained_var_fit,
            "retained_var_new": self.retained_var_new,
            "retained_var_drop": self.retained_var_drop,
            "stale": self.stale,
        }


class TQEIndex:
    """A persisted, compressed Track-1 search index with a full lifecycle."""

    def __init__(
        self,
        pca: PCAMatryoshka,
        *,
        bits: int,
        seed: int,
        rotation: str,
        metric: str,
        fit_retained_var: float,
        format_version: int = CURRENT_VERSION,
    ):
        self._pca = pca
        self._bits = int(bits)
        self._seed = int(seed)
        self._rotation = rotation
        self._metric = metric
        self._fit_retained_var = float(fit_retained_var)
        self._format_version = format_version
        self._created_utc = _utc_now()

        self._pipeline = pca.with_quantizer(bits=bits, seed=seed, rotation=rotation)
        self._adc = ADCIndex(self._pipeline)
        # Row-parallel state.
        self._ids = np.zeros(0, dtype=np.int64)
        self._tomb = np.zeros(0, dtype=np.uint8)
        self._originals: np.ndarray | None = None
        self._next_id = 0
        self._id_pos: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        embeddings: np.ndarray,
        *,
        output_dim: int | None = None,
        bits: int = 3,
        seed: int = 42,
        rotation: str = "qr",
        whiten: bool = False,
        metric: str = "cosine",
        keep_originals: bool = True,
        ids: np.ndarray | None = None,
        train_cap: int = 200_000,
    ) -> TQEIndex:
        """Fit the PCA basis on ``embeddings`` and build the index."""
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (n, dim), got {x.shape}")
        n, dim = x.shape
        out = dim if output_dim is None else min(int(output_dim), dim)
        pca = PCAMatryoshka(input_dim=dim, output_dim=out, whiten=whiten)
        fit = pca.fit(x[: min(n, train_cap)])
        retained = float(fit.total_variance_explained)
        idx = cls(
            pca,
            bits=bits,
            seed=seed,
            rotation=rotation,
            metric=metric,
            fit_retained_var=retained,
        )
        idx._append(x, ids, keep_originals=keep_originals)
        return idx

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Write the index to ``path`` (atomic, CRC-checked container)."""
        from . import __version__  # local: avoid an __init__ circular import

        arrays: dict[str, dict] = {}
        sections: list[tuple[str, bytes]] = []

        def add_array(name: str, a: np.ndarray) -> None:
            blob, spec = _arr_bytes(a)
            arrays[name] = spec
            sections.append((name, blob))

        # PCA basis (enough to recompress queries + new vectors, no refit).
        add_array("pca_mean", self._pca._mean)
        add_array("pca_components", self._pca._components)
        add_array("pca_eigenvalues", self._pca._eigenvalues)
        add_array("pca_all_eigenvalues", self._pca._all_eigenvalues)
        # The ADC payload.
        add_array("codes", self._adc._codes)
        add_array("cnorm", self._adc._cnorm)
        add_array("vrnorm", self._adc._vrnorm)
        if self._format_version >= 2:
            add_array("ids", self._ids)
            add_array("tombstones", self._tomb)
        has_originals = self._originals is not None
        if has_originals:
            add_array("originals", self._originals)

        meta = {
            "format_version": self._format_version,
            "tool_version": __version__,
            "created_utc": self._created_utc,
            "metric": self._metric,
            "pca": {
                "input_dim": int(self._pca.input_dim),
                "output_dim": int(self._pca.output_dim),
                "whiten": bool(self._pca.whiten),
            },
            "quant": {
                "bits": self._bits,
                "seed": self._seed,
                "rotation": self._rotation,
            },
            "n_rows": int(self._adc.size),
            "n_live": int(self.n_live),
            "next_id": int(self._next_id),
            "fit_retained_var": self._fit_retained_var,
            "has_originals": has_originals,
            "arrays": arrays,
        }
        meta_blob = json.dumps(meta).encode("utf-8")
        # meta first so a reader can learn the layout up front.
        write_container(path, self._format_version, [("meta", meta_blob), *sections])

    @classmethod
    def open(cls, path: str) -> TQEIndex:
        """Load an index from ``path`` (verifies every section's CRC32)."""
        version, sections = read_container(path)
        if version not in _SUPPORTED:
            raise ValueError(
                f"unsupported index format version {version}; "
                f"supported: {_SUPPORTED}"
            )
        meta = json.loads(sections["meta"].decode("utf-8"))
        arrays = meta["arrays"]

        def arr(name: str) -> np.ndarray:
            return _arr_from(sections[name], arrays[name])

        pca = PCAMatryoshka(
            input_dim=meta["pca"]["input_dim"],
            output_dim=meta["pca"]["output_dim"],
            whiten=meta["pca"]["whiten"],
        )
        pca._mean = arr("pca_mean")
        pca._components = arr("pca_components")
        pca._eigenvalues = arr("pca_eigenvalues")
        pca._all_eigenvalues = arr("pca_all_eigenvalues")

        idx = cls(
            pca,
            bits=meta["quant"]["bits"],
            seed=meta["quant"]["seed"],
            rotation=meta["quant"]["rotation"],
            metric=meta["metric"],
            fit_retained_var=meta.get("fit_retained_var", 0.0),
            format_version=meta["format_version"],
        )
        idx._created_utc = meta.get("created_utc", idx._created_utc)
        # Restore the ADC payload directly — no recompute.
        idx._adc._codes = np.ascontiguousarray(arr("codes"))
        idx._adc._cnorm = arr("cnorm").copy()
        idx._adc._vrnorm = arr("vrnorm").copy()
        n_rows = idx._adc.size
        if version >= 2:
            idx._ids = arr("ids").copy()
            idx._tomb = arr("tombstones").copy()
        else:
            # v1: ids were implicit positional, no tombstones.
            idx._ids = np.arange(n_rows, dtype=np.int64)
            idx._tomb = np.zeros(n_rows, dtype=np.uint8)
        idx._originals = arr("originals").copy() if meta["has_originals"] else None
        idx._next_id = int(meta.get("next_id", n_rows))
        idx._reindex_ids()
        return idx

    # ------------------------------------------------------------------ #
    # Mutations                                                          #
    # ------------------------------------------------------------------ #
    def _append(
        self,
        x: np.ndarray,
        ids: np.ndarray | None,
        *,
        keep_originals: bool,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        m = len(x)
        if ids is None:
            new_ids = np.arange(self._next_id, self._next_id + m, dtype=np.int64)
        else:
            new_ids = np.asarray(ids, dtype=np.int64)
            if len(new_ids) != m:
                raise ValueError("ids length must match number of embeddings")
            dup = set(new_ids.tolist()) & set(self._id_pos)
            if dup:
                raise ValueError(f"ids already present: {sorted(dup)[:5]}")
        self._adc.add(x)
        self._ids = np.concatenate([self._ids, new_ids])
        self._tomb = np.concatenate([self._tomb, np.zeros(m, dtype=np.uint8)])
        # Originals are all-or-nothing for the index's lifetime.
        if keep_originals:
            self._originals = (
                x.copy()
                if self._originals is None
                else np.concatenate([self._originals, x])
            )
        if m:
            self._next_id = max(self._next_id, int(new_ids.max()) + 1)
        self._reindex_ids()
        return new_ids

    def add(self, embeddings: np.ndarray, ids: np.ndarray | None = None) -> np.ndarray:
        """Append new vectors, compressed by the existing basis; returns ids."""
        keep = self._originals is not None
        x = np.asarray(embeddings, dtype=np.float32)
        return self._append(x, ids, keep_originals=keep)

    def delete(self, ids) -> int:
        """Tombstone the rows for ``ids``. Returns how many were newly deleted."""
        deleted = 0
        for i in np.atleast_1d(np.asarray(ids, dtype=np.int64)):
            pos = self._id_pos.get(int(i))
            if pos is not None and not self._tomb[pos]:
                self._tomb[pos] = 1
                deleted += 1
        return deleted

    def compact(self) -> int:
        """Physically drop tombstoned rows. Returns rows reclaimed."""
        live = self._tomb == 0
        reclaimed = int((~live).sum())
        if reclaimed == 0:
            return 0
        self._adc._codes = np.ascontiguousarray(self._adc._codes[live])
        self._adc._cnorm = self._adc._cnorm[live].copy()
        self._adc._vrnorm = self._adc._vrnorm[live].copy()
        self._ids = self._ids[live].copy()
        self._tomb = np.zeros(len(self._ids), dtype=np.uint8)
        if self._originals is not None:
            self._originals = self._originals[live].copy()
        self._reindex_ids()
        return reclaimed

    def migrate(self, to_version: int) -> None:
        """Upgrade the on-disk format version in place."""
        if to_version not in _SUPPORTED:
            raise ValueError(f"unknown target version {to_version}; {_SUPPORTED}")
        if to_version < self._format_version:
            raise ValueError(f"cannot downgrade {self._format_version} -> {to_version}")
        # v1 -> v2 materializes explicit ids + tombstones; both already exist in
        # memory (open() synthesized them for v1), so the upgrade is a version bump
        # whose effect is that save() now writes those sections.
        self._format_version = to_version

    # ------------------------------------------------------------------ #
    # Queries                                                            #
    # ------------------------------------------------------------------ #
    def search(
        self, queries: np.ndarray, k: int = 10, rerank: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` external ids per query, excluding tombstoned rows.

        With ``rerank > 0`` the top ``k * rerank`` ADC candidates are rescored by
        exact inner product (against stored originals if kept, else the
        compressed reconstruction) — the high-recall two-stage path.
        """
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None]
        n_rows = self._adc.size
        if n_rows == 0:
            raise RuntimeError("index is empty")
        n_dead = int(self._tomb.sum())
        # Over-fetch so that after dropping tombstones we still have k * rerank.
        want = k * max(rerank, 1)
        fetch = min(n_rows, want + n_dead)
        cand_pos, cand_sc = self._adc.search(q, k=fetch)

        out_ids = np.full((len(q), k), -1, dtype=np.int64)
        out_sc = np.full((len(q), k), np.nan, dtype=np.float32)
        rr_src = self._originals if (rerank and self._originals is not None) else None
        recon = None
        if rerank and rr_src is None:
            recon = self._reconstruct_all()  # approximate rerank basis
        basis = rr_src if rr_src is not None else recon
        # Rerank in the index's own metric (cosine by default), not raw dot —
        # the corpus need not be unit-norm.
        qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
        for r in range(len(q)):
            pos = cand_pos[r]
            sc = cand_sc[r]
            keep = pos[(pos >= 0) & (self._tomb[pos] == 0)]
            ksc = sc[(pos >= 0) & (self._tomb[pos] == 0)]
            if rerank and len(keep):
                cand = basis[keep]
                if self._metric == "l2":
                    exact = -((cand - q[r]) ** 2).sum(axis=1)
                else:  # cosine
                    cn = cand / np.maximum(
                        np.linalg.norm(cand, axis=1, keepdims=True), 1e-30
                    )
                    exact = cn @ qn[r]
                order = np.argsort(-exact)[:k]
                sel = keep[order]
                sels = exact[order].astype(np.float32)
            else:
                sel = keep[:k]
                sels = ksc[:k]
            out_ids[r, : len(sel)] = self._ids[sel]
            out_sc[r, : len(sels)] = sels
        return out_ids, out_sc

    def certify(
        self, sample: int = 512, n_anchors: int = 200, seed: int = 0
    ) -> RankCertificate:
        """Distribution-free rank certificate over a sample of live vectors.

        Requires stored originals (``keep_originals=True`` at create): the
        certificate compares the exact ranking to the compressed reconstruction's
        ranking. Acceptance is the Kendall-tau / Spearman floor, never cosine.
        """
        if self._originals is None:
            raise ValueError(
                "certify needs the original vectors; rebuild with keep_originals=True"
            )
        live = np.flatnonzero(self._tomb == 0)
        if len(live) == 0:
            raise RuntimeError("no live vectors to certify")
        rng = np.random.default_rng(seed)
        if len(live) <= sample:
            take = live
        else:
            take = rng.choice(live, size=sample, replace=False)
        orig = self._originals[take]
        codes = self._pipeline.compress_batch(orig)
        recon = np.asarray(self._pipeline.decompress_batch(codes))
        return certificate_from_embeddings(
            orig,
            recon,
            n_anchors=min(n_anchors, len(take)),
            metric=self._metric,
            seed=seed,
        )

    def drift(
        self, embeddings: np.ndarray, var_drop_threshold: float = 0.05
    ) -> DriftReport:
        """Is the stored PCA basis still a good fit for ``embeddings``?

        Compares the variance fraction the basis retains on the new batch to what
        it retained at fit time, plus the shift of the batch mean. A large drop
        means the basis is stale — recompress (recreate) before recall silently
        erodes.
        """
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self._pca.input_dim:
            raise ValueError(
                f"embeddings must be (n, {self._pca.input_dim}), got {x.shape}"
            )
        centered = x - self._pca._mean
        total_var = float((centered**2).sum(axis=1).mean())
        proj = centered @ self._pca._components.T  # (n, out)
        kept_var = float((proj**2).sum(axis=1).mean())
        retained_new = kept_var / max(total_var, 1e-12)
        mean_shift = float(
            np.linalg.norm(x.mean(axis=0) - self._pca._mean)
            / max(np.linalg.norm(self._pca._mean), 1e-12)
        )
        drop = self._fit_retained_var - retained_new
        return DriftReport(
            mean_shift=mean_shift,
            retained_var_fit=self._fit_retained_var,
            retained_var_new=retained_new,
            retained_var_drop=drop,
            stale=bool(drop > var_drop_threshold),
        )

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    @property
    def n_live(self) -> int:
        return int((self._tomb == 0).sum())

    @property
    def n_rows(self) -> int:
        return int(self._adc.size)

    def stats(self) -> dict:
        bytes_per_vec = self._adc._codes.shape[1] if self._adc.size else 0
        return {
            "format_version": self._format_version,
            "n_rows": self.n_rows,
            "n_live": self.n_live,
            "n_tombstoned": self.n_rows - self.n_live,
            "input_dim": int(self._pca.input_dim),
            "output_dim": int(self._pca.output_dim),
            "bits": self._bits,
            "code_bytes_per_vec": int(bytes_per_vec),
            "has_originals": self._originals is not None,
            "compression_ratio": round(float(self._pipeline.compression_ratio), 3),
            "metric": self._metric,
            "fit_retained_var": self._fit_retained_var,
        }

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _reindex_ids(self) -> None:
        self._id_pos = {int(i): p for p, i in enumerate(self._ids)}

    def _reconstruct_all(self) -> np.ndarray:
        """Reconstruct every stored vector into the input space (whiten=False)."""
        if self._pca.whiten:
            raise ValueError(
                "reconstruction-from-codes needs whiten=False; "
                "rebuild with keep_originals=True for exact rerank under whitening"
            )
        cc = self._adc._cent[self._adc._codes]  # (N, out) rotated unit dirs
        unrot = self._pipeline.quantizer._unrotate(cc)
        xp = self._adc._cnorm[:, None] * unrot
        return np.asarray(self._pca.inverse_transform(xp), dtype=np.float32)


def index_info(path: str) -> dict:
    """Cheap header/section summary without loading payloads (``tqp index info``)."""
    version, refs = read_directory(path)
    return {
        "path": path,
        "container_version": version,
        "sections": {r.name: {"length": r.length, "crc32": r.crc32} for r in refs},
    }
