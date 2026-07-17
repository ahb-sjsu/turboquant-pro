"""Production index lifecycle: ingest -> search -> update -> compact -> migrate
-> certify -> drift, plus persistence round-trips.

Pure-numpy; runs in CI. Acceptance is rank fidelity (recall / the certificate),
never reconstruction cosine.
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import TQEIndex
from turboquant_pro.index import CURRENT_VERSION


def _corpus(n=1500, dim=64, seed=0):
    """Moderate-rank embeddings with a decaying spectrum — a stand-in corpus."""
    rng = np.random.default_rng(seed)
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    scale = np.linspace(1.0, 0.3, rank)
    coeffs = rng.standard_normal((n, rank)) * scale
    return (coeffs @ basis + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)


def _exact_topk(queries, base, k):
    bn = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-30)
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-30)
    return np.argsort(-(qn @ bn.T), axis=1)[:, :k]


def _recall(idx, corpus, queries, k=10, rerank=0):
    gt = _exact_topk(queries, corpus, k)  # ids == positions for a fresh index
    got, _ = idx.search(queries, k=k, rerank=rerank)
    return float(np.mean([len(set(a) & set(g)) / k for a, g in zip(got, gt)]))


# --------------------------------------------------------------------------- #
# Persistence round-trip                                                      #
# --------------------------------------------------------------------------- #


def test_create_save_open_roundtrip(tmp_path):
    corpus = _corpus()
    idx = TQEIndex.create(corpus, output_dim=32, bits=4, seed=1)
    p = tmp_path / "a.tqe"
    idx.save(str(p))
    re = TQEIndex.open(str(p))
    assert re.stats()["n_rows"] == len(corpus)
    assert re.stats()["format_version"] == CURRENT_VERSION
    q = corpus[:50]
    a, _ = idx.search(q, k=10)
    b, _ = re.search(q, k=10)
    np.testing.assert_array_equal(a, b)  # reopened index searches identically


def test_search_recall_is_reasonable(tmp_path):
    corpus = _corpus()
    idx = TQEIndex.create(corpus, output_dim=64, bits=4, seed=1)
    q = corpus[:100]
    assert _recall(idx, corpus, q, k=10) > 0.5
    # Exact rerank (stored originals) recovers near-perfect recall.
    assert _recall(idx, corpus, q, k=10, rerank=10) > 0.95


# --------------------------------------------------------------------------- #
# Append                                                                       #
# --------------------------------------------------------------------------- #


def test_add_appends_and_assigns_ids(tmp_path):
    a, b = _corpus(1000), _corpus(500, seed=9)
    idx = TQEIndex.create(a, output_dim=32, bits=4)
    new_ids = idx.add(b)
    assert idx.n_rows == 1500
    assert list(new_ids) == list(range(1000, 1500))
    # a vector from the appended batch is findable by its own content
    got, _ = idx.search(b[:5], k=5)
    assert all(any(i >= 1000 for i in row) for row in got)


# --------------------------------------------------------------------------- #
# Delete + compact                                                             #
# --------------------------------------------------------------------------- #


def test_delete_tombstones_excluded_from_search(tmp_path):
    corpus = _corpus(800)
    idx = TQEIndex.create(corpus, output_dim=32, bits=4)
    # delete the true nearest neighbours of a query, then confirm they vanish
    q = corpus[:20]
    gt = _exact_topk(q, corpus, 5)
    victims = np.unique(gt.ravel())
    idx.delete(victims)
    assert idx.n_live == 800 - len(victims)
    got, _ = idx.search(q, k=10)
    assert not (set(victims.tolist()) & set(got.ravel().tolist()))


def test_compact_reclaims_and_preserves_live(tmp_path):
    corpus = _corpus(800)
    idx = TQEIndex.create(corpus, output_dim=32, bits=4)
    victims = np.arange(700, 750)
    idx.delete(victims)
    reclaimed = idx.compact()
    assert reclaimed == 50
    assert idx.n_rows == 750 and idx.n_live == 750
    # External ids are preserved (not renumbered) through compaction, deleted ids
    # are gone, and each surviving query still finds itself (exact rerank).
    q_ids = np.arange(100, 150)
    got, _ = idx.search(corpus[q_ids], k=10, rerank=10)
    assert not (set(victims.tolist()) & set(got.ravel().tolist()))
    assert all(q_ids[i] in got[i] for i in range(len(q_ids)))


def test_compact_noop_without_tombstones(tmp_path):
    idx = TQEIndex.create(_corpus(300), output_dim=32, bits=4)
    assert idx.compact() == 0


# --------------------------------------------------------------------------- #
# Migrate                                                                      #
# --------------------------------------------------------------------------- #


def test_migrate_v1_to_v2(tmp_path):
    corpus = _corpus(400)
    idx = TQEIndex.create(corpus, output_dim=32, bits=4)
    # Write a legacy v1 (implicit ids, no tombstone section).
    idx._format_version = 1
    p = tmp_path / "legacy.tqe"
    idx.save(str(p))
    v1 = TQEIndex.open(str(p))
    assert v1.stats()["format_version"] == 1
    v1.migrate(2)
    p2 = tmp_path / "upgraded.tqe"
    v1.save(str(p2))
    v2 = TQEIndex.open(str(p2))
    assert v2.stats()["format_version"] == 2
    # v2 unlocks delete/compact; ids are preserved through the upgrade
    assert v2.delete([0, 1, 2]) == 3
    assert v2.compact() == 3


def test_migrate_rejects_downgrade(tmp_path):
    idx = TQEIndex.create(_corpus(200), output_dim=16, bits=4)
    with pytest.raises(ValueError):
        idx.migrate(1)


# --------------------------------------------------------------------------- #
# Certify                                                                      #
# --------------------------------------------------------------------------- #


def test_certify_nonvacuous(tmp_path):
    corpus = _corpus(1000)
    idx = TQEIndex.create(corpus, output_dim=64, bits=4)
    cert = idx.certify(sample=300, n_anchors=128, seed=0)
    assert not cert.vacuous, cert.as_dict()
    assert cert.tau_floor > 0.0


def test_certify_requires_originals(tmp_path):
    idx = TQEIndex.create(_corpus(300), output_dim=32, bits=4, keep_originals=False)
    with pytest.raises(ValueError):
        idx.certify()


def test_no_originals_search_still_works(tmp_path):
    """Without stored originals, single-pass search works and rerank degrades
    to reconstruction (still runs, exact-rerank precision not guaranteed)."""
    corpus = _corpus(600)
    idx = TQEIndex.create(corpus, output_dim=32, bits=4, keep_originals=False)
    got, _ = idx.search(corpus[:10], k=10)
    assert got.shape == (10, 10)
    got_rr, _ = idx.search(corpus[:10], k=10, rerank=5)  # reconstruction rerank
    assert got_rr.shape == (10, 10)


# --------------------------------------------------------------------------- #
# Drift                                                                        #
# --------------------------------------------------------------------------- #


def test_drift_flags_distribution_shift(tmp_path):
    # Same distribution = a held-out split of one generation (shared subspace).
    big = _corpus(1300, dim=64, seed=0)
    idx = TQEIndex.create(big[:1000], output_dim=32, bits=4)
    assert not idx.drift(big[1000:]).stale
    # A full-rank isotropic batch the 32-d basis cannot capture: stale.
    rng = np.random.default_rng(7)
    shifted = (rng.standard_normal((300, 64)) * 3.0 + 5.0).astype(np.float32)
    report = idx.drift(shifted)
    assert report.retained_var_new < report.retained_var_fit
    assert report.stale


# --------------------------------------------------------------------------- #
# Full lifecycle (the exit criterion)                                         #
# --------------------------------------------------------------------------- #


def test_full_lifecycle(tmp_path):
    """ingest -> search -> update -> compact -> migrate -> certify -> monitor."""
    corpus = _corpus(1200)
    p = tmp_path / "life.tqe"

    # ingest
    idx = TQEIndex.create(corpus, output_dim=48, bits=4, seed=3)
    idx.save(str(p))

    # search
    idx = TQEIndex.open(str(p))
    q = corpus[:60]
    assert _recall(idx, corpus, q, k=10, rerank=10) > 0.9

    # update: append + delete
    extra = _corpus(300, seed=5)
    idx.add(extra)
    assert idx.n_rows == 1500
    idx.delete(np.arange(0, 100))
    idx.save(str(p))

    # compact
    idx = TQEIndex.open(str(p))
    assert idx.compact() == 100
    idx.save(str(p))

    # certify
    idx = TQEIndex.open(str(p))
    cert = idx.certify(sample=300, n_anchors=128)
    assert not cert.vacuous

    # monitor: still healthy after the whole lifecycle
    assert idx.n_live == 1400
    assert idx.stats()["n_tombstoned"] == 0


# --------------------------------------------------------------------------- #
# Memmap + blocked search (bounded-memory, large-index path)                  #
# --------------------------------------------------------------------------- #


def test_mmap_search_matches_in_ram(tmp_path):
    corpus = _corpus(1500)
    idx = TQEIndex.create(corpus, output_dim=32, bits=4, seed=1)
    p = tmp_path / "m.tqe"
    idx.save(str(p))
    ram = TQEIndex.open(str(p))
    mm = TQEIndex.open(str(p), mmap=True)
    assert mm._mmap and isinstance(mm._adc._codes, np.memmap)
    assert not ram._mmap
    q = corpus[:50]
    # Exact rerank is deterministic — memmap and in-RAM must agree exactly.
    a_ids, _ = ram.search(q, k=10, rerank=10)
    b_ids, _ = mm.search(q, k=10, rerank=10)
    np.testing.assert_array_equal(a_ids, b_ids)
    # Single-pass top-k sets agree too (same ADC scores).
    a1, _ = ram.search(q, k=10)
    b1, _ = mm.search(q, k=10)
    assert all(set(x) == set(y) for x, y in zip(a1, b1))


def test_blocked_search_matches_full(tmp_path):
    corpus = _corpus(1200)
    idx = TQEIndex.create(corpus, output_dim=32, bits=4)
    q = corpus[:60]
    full, _ = idx.search(q, k=10, rerank=10)
    blocked, _ = idx.search(q, k=10, rerank=10, block=131)  # tiny block, many chunks
    np.testing.assert_array_equal(full, blocked)


def test_mmap_is_read_only(tmp_path):
    corpus = _corpus(400)
    idx = TQEIndex.create(corpus, output_dim=24, bits=4)
    p = tmp_path / "ro.tqe"
    idx.save(str(p))
    mm = TQEIndex.open(str(p), mmap=True)
    import pytest as _pytest

    for op in (
        lambda: mm.add(corpus[:10]),
        lambda: mm.delete([0]),
        lambda: mm.compact(),
        lambda: mm.migrate(2),
    ):
        with _pytest.raises(RuntimeError):
            op()


def test_mmap_certify_drift_and_delete_exclusion(tmp_path):
    corpus = _corpus(1000)
    idx = TQEIndex.create(corpus, output_dim=32, bits=4)
    # tombstone some rows before saving, then reopen memory-mapped
    idx.delete(np.arange(0, 40))
    p = tmp_path / "c.tqe"
    idx.save(str(p))
    mm = TQEIndex.open(str(p), mmap=True)
    # tombstoned ids are still excluded from a memmap search
    got, _ = mm.search(corpus[:20], k=10)
    assert not (set(range(40)) & set(got.ravel().tolist()))
    # certify + drift work over memmapped arrays
    assert not mm.certify(sample=200, n_anchors=100).vacuous
    assert not mm.drift(corpus[500:700]).stale
    assert mm.n_live == 960 and mm.stats()["n_rows"] == 1000
