# TurboQuant Pro: STRATA Phases 3-4 — per-area operating points & blast radius
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""STRATA Phase 3 (per-area operating points, identity-gated) and the
Phase-4 normative core (area-scoped contract keys + stale sets).

Phase 3 (docs/STRATA_RFC.md §4): capacity is spent by MEASURED FRAGILITY
(per-stratum hubdiff), not variance alone. Record metadata gains
``{area_map_digest, area_id, area_codec_params}``; a searcher holding a
different ``area_map_digest`` MUST refuse — enumerate, don't decode.

Phase 4 (§5): the recall-contract key extends with (area_map_digest,
area_id); every mutation class has a documented worst-case stale set.
Database wiring (pgvector catalog columns) is 2.2-class and lands there —
the semantics here are the normative part.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .strata import AreaMap, require_same_map


def allocate_by_fragility(
    hubdiff_report: dict,
    area_counts: dict[str, int],
    *,
    bit_options: tuple[int, ...] = (2, 3, 4),
    budget_bits_per_row: float = 3.0,
) -> dict[str, int]:
    """Assign per-area bit widths by measured fragility under a global budget.

    Fragility = per-stratum anti-hub recall from a ``tqp-strata-report/1``
    hubdiff report (the §4 allocator: spend on the measured weak, not the
    high-variance). Greedy: everyone starts at the floor; upgrades go
    most-fragile-first while the row-weighted mean stays within budget.
    ABSTAIN strata get the budget median (no verdict ⇒ no special spend).
    """
    floor, ceil = min(bit_options), max(bit_options)
    frag = {}
    for a in hubdiff_report["areas"]:
        if a["verdict"] != "ABSTAIN" and a.get("anti_hub_recall") is not None:
            frag[a["id"]] = a["anti_hub_recall"]
    bits = {area: floor for area in area_counts}
    total = sum(area_counts.values())

    def mean_bits():
        return sum(bits[a] * area_counts[a] for a in bits) / max(total, 1)

    for area in sorted(frag, key=frag.get):  # most fragile first
        for step in sorted(o for o in bit_options if o > floor):
            old = bits[area]
            bits[area] = step
            if mean_bits() > budget_bits_per_row:
                bits[area] = old
                break
    mid = sorted(bit_options)[len(bit_options) // 2]
    for area in bits:
        if area not in frag and bits[area] == floor:
            bits[area] = min(mid, ceil)
            if mean_bits() > budget_bits_per_row:
                bits[area] = floor
    return bits


class StratifiedIndex:
    """Per-area TQE indexes at per-area operating points, identity-gated.

    Every constituent index is tagged with ``{area_map_digest, area_id,
    area_codec_params}``. ``search`` REQUIRES the caller's map digest and
    refuses on mismatch — cross-map reads are cache misses, never
    best-effort (TQE1 §5/§8 rule, inherited).
    """

    def __init__(self, area_map: AreaMap):
        self._map = area_map
        self._parts: dict[str, Any] = {}
        self._row_ids: dict[str, np.ndarray] = {}
        self._params: dict[str, dict] = {}

    @classmethod
    def build(
        cls,
        base: np.ndarray,
        area_map: AreaMap,
        bits_by_area: dict[str, int],
        *,
        output_dim: int | None = None,
        seed: int = 42,
    ) -> StratifiedIndex:
        from .index import TQEIndex

        x = np.ascontiguousarray(base, dtype=np.float32)
        lab = np.asarray(area_map.labels)
        idx = cls(area_map)
        for area in area_map.areas:
            rows = np.where(lab == area)[0]
            if len(rows) < 8:  # too thin to fit a quantizer: store exact ids only
                continue
            bits = bits_by_area.get(area, 3)
            part = TQEIndex.create(
                x[rows],
                output_dim=output_dim,
                bits=bits,
                seed=seed,
                keep_originals=False,
                ids=rows.astype(np.int64),
            )
            idx._parts[area] = part
            idx._row_ids[area] = rows
            idx._params[area] = {
                "area_map_digest": area_map.digest,
                "area_id": area,
                "area_codec_params": {"bits": bits, "output_dim": output_dim},
            }
        return idx

    @property
    def areas(self) -> tuple[str, ...]:
        return tuple(sorted(self._parts))

    def metadata(self, area: str) -> dict:
        return dict(self._params[area])

    def bits_per_row_mean(self) -> float:
        tot = sum(len(r) for r in self._row_ids.values())
        return sum(
            self._params[a]["area_codec_params"]["bits"] * len(self._row_ids[a])
            for a in self._parts
        ) / max(tot, 1)

    def search(
        self, queries: np.ndarray, k: int, *, area_map_digest: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Global top-k by merging per-area searches. Digest-gated."""
        require_same_map(area_map_digest, self._map.digest)
        q = np.asarray(queries, dtype=np.float32)
        all_ids, all_sc = [], []
        for area in self.areas:
            ids, sc = self._parts[area].search(q, k=k)
            all_ids.append(ids)
            all_sc.append(np.nan_to_num(sc, nan=-np.inf))
        ids = np.concatenate(all_ids, axis=1)
        sc = np.concatenate(all_sc, axis=1)
        order = np.argsort(-sc, axis=1)[:, :k]
        return (
            np.take_along_axis(ids, order, 1),
            np.take_along_axis(sc, order, 1),
        )


# ----------------------------------------------------------------------
# Phase 4 normative core: area-scoped contract keys + stale sets (§5).
# ----------------------------------------------------------------------


def area_scoped_contract_key(
    base_key: dict, area_map_digest: str, area_id: str
) -> dict:
    """Extend a recall-contract catalog key with the §5 area dimensions."""
    out = dict(base_key)
    out["area_map_digest"] = area_map_digest
    out["area_id"] = area_id
    return out


def stale_set(
    event: str,
    area_map: AreaMap,
    *,
    area: str | None = None,
    adjacency: dict[str, set[str]] | None = None,
) -> set[str]:
    """§5 staleness: which area certificates a mutation invalidates.

    ``mutation`` (insert/delete/update in one area, fixed digest) stales
    that area plus its overlap/adjacency neighbours; ``map_recompute``
    stales everything; ``operating_point`` stales exactly the changed
    area. A guarantee that can go stale silently is not a guarantee — this
    is the bounded, NAMED blast radius.
    """
    areas = set(area_map.areas)
    if event == "map_recompute":
        return areas
    if area is None or area not in areas:
        raise KeyError(f"unknown area {area!r} for event {event!r}")
    if event == "mutation":
        nbrs = (adjacency or {}).get(area, set())
        return {area} | (set(nbrs) & areas)
    if event == "operating_point":
        return {area}
    raise KeyError(f"unregistered mutation class {event!r}")
