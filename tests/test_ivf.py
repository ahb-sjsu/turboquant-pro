"""IVF coarse-partition layer: sublinear probing that reproduces the brute-force
ADC ranking. Validates the admissible early stop (probe-all == exact), high recall
at a small scan fraction, and the A*-style adaptive stop.
"""

from __future__ import annotations

import numpy as np

from turboquant_pro import ADCIndex, IVFIndex, PCAMatryoshka


def _corpus(n=4000, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    coeffs = rng.standard_normal((n, rank)) * np.linspace(1.0, 0.3, rank)
    return (coeffs @ basis + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)


def _brute_adc_topk(corpus, queries, k, out_dim=32, bits=4):
    pca = PCAMatryoshka(input_dim=corpus.shape[1], output_dim=out_dim)
    pca.fit(corpus)
    adc = ADCIndex(pca.with_quantizer(bits=bits)).add(corpus)
    idx, _ = adc.search(queries, k=k)
    return idx


def _recall(got, ref, k):
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(got[:, :k], ref)]))


def test_probe_all_equals_bruteforce():
    # Probing every cell + scoring all candidates is exactly the brute ADC scan.
    corpus = _corpus()
    q = corpus[:60]
    ref = _brute_adc_topk(corpus, q, 10)
    ivf = IVFIndex.create(corpus, output_dim=32, bits=4)
    ids, _, stats = ivf.search(q, k=10, nprobe=ivf.stats()["nlist"], return_stats=True)
    assert _recall(ids, ref, 10) == 1.0  # identical ranking, no approximation
    assert all(s.scan_fraction == 1.0 for s in stats)


def test_high_recall_at_small_scan_fraction():
    corpus = _corpus()
    q = corpus[:80]
    ref = _brute_adc_topk(corpus, q, 10)
    ivf = IVFIndex.create(corpus, output_dim=32, bits=4)
    nlist = ivf.stats()["nlist"]
    # Probing a quarter of the cells (best-first) recovers most neighbours.
    ids, _, stats = ivf.search(q, k=10, nprobe=max(8, nlist // 4), return_stats=True)
    frac = float(np.mean([s.scan_fraction for s in stats]))
    assert _recall(ids, ref, 10) > 0.8
    assert frac < 0.4  # sublinear: touches well under half the corpus


def test_adaptive_weighted_stop_is_sublinear():
    corpus = _corpus()
    q = corpus[:80]
    ref = _brute_adc_topk(corpus, q, 10)
    ivf = IVFIndex.create(corpus, output_dim=32, bits=4)
    # weighted A*: shrinking the radius prunes more. beta<1 scans less than exact.
    ids, _, stats = ivf.search(
        q, k=10, nprobe=None, bound="weighted", radius_scale=0.5, return_stats=True
    )
    exact_frac = np.mean(
        [s.scan_fraction for s in ivf.search(
            q, k=10, nprobe=None, bound="admissible", return_stats=True)[2]]
    )
    frac = float(np.mean([s.scan_fraction for s in stats]))
    assert _recall(ids, ref, 10) > 0.6  # keeps useful recall
    assert frac < exact_frac  # prunes strictly more than the admissible bound
    assert all(s.cells_probed >= 1 for s in stats)


def test_admissible_stop_is_exact():
    # The admissible bound never prunes a cell that could contain a better point,
    # so its result matches probing everything (coarse-exact), even if it scans a lot.
    corpus = _corpus(2500)
    q = corpus[:50]
    ivf = IVFIndex.create(corpus, output_dim=32, bits=4)
    exact, _ = ivf.search(q, k=10, nprobe=ivf.stats()["nlist"])
    adm, _ = ivf.search(q, k=10, nprobe=None, bound="admissible")
    assert _recall(adm, exact, 10) > 0.99  # admissible stop == full probe


def test_rerank_finds_self():
    corpus = _corpus(3000)
    ivf = IVFIndex.create(corpus, output_dim=32, bits=4, keep_originals=True)
    q = corpus[:40]
    ids, _ = ivf.search(q, k=10, nprobe=None, bound="admissible", rerank=10)
    assert all(i in ids[i] for i in range(len(q)))  # each row retrieves itself


def test_stats_and_partition_sane():
    corpus = _corpus(2000)
    ivf = IVFIndex.create(corpus, output_dim=32, bits=4, nlist=40)
    st = ivf.stats()
    assert st["nlist"] == 40
    assert st["n_rows"] == 2000
    assert st["cell_min"] >= 0 and st["cell_max"] <= 2000
    assert 0.0 <= st["radius_mean_deg"] <= 180.0
