"""Tests for ADCIndex (numpy fallback path; kernel path if compiled)."""

import numpy as np
import pytest

from turboquant_pro import ADCIndex, PCAMatryoshka


def _exact_top(q, C, k):
    return np.argsort(-(q @ C.T), axis=1)[:, :k]


def test_adc_index_basic():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 128)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pca = PCAMatryoshka(input_dim=128, output_dim=64)
    pca.fit(X[:1000])
    index = ADCIndex(pca.with_quantizer(bits=3)).add(X)
    assert index.size == 2000
    idx, sc = index.search(X[:10], k=5)
    assert idx.shape == (10, 5)
    assert sc.shape == (10, 5)
    assert idx.min() >= 0 and idx.max() < 2000


def test_adc_index_add_accumulates():
    # add() must append, not replace: index.add(a).add(b) holds both batches.
    rng = np.random.default_rng(7)
    X = rng.standard_normal((1000, 64)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pca = PCAMatryoshka(input_dim=64, output_dim=32)
    pca.fit(X[:500])

    one_shot = ADCIndex(pca.with_quantizer(bits=4)).add(X)
    incremental = ADCIndex(pca.with_quantizer(bits=4)).add(X[:400]).add(X[400:])

    assert one_shot.size == 1000
    assert incremental.size == 1000  # both batches retained, not just the last
    # Incremental indexing reproduces the one-shot codes exactly (same order).
    np.testing.assert_array_equal(one_shot._codes, incremental._codes)


def test_adc_index_recall_reasonable():
    # On low-dim Gaussian data the compressed ADC should recover most neighbors.
    rng = np.random.default_rng(3)
    X = rng.standard_normal((3000, 64)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pca = PCAMatryoshka(input_dim=64, output_dim=48)
    pca.fit(X[:1500])
    index = ADCIndex(pca.with_quantizer(bits=4)).add(X)
    q = X[:200]
    gt = _exact_top(q, X, 10)
    # with exact rerank, recall@10 should be high
    pred = index.search(q, k=10, rerank=10, originals=X)
    rec = np.mean([len(set(gt[i]) & set(pred[i])) / 10 for i in range(len(q))])
    assert rec > 0.6


def test_adc_index_rerank_shape():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1500, 96)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pca = PCAMatryoshka(input_dim=96, output_dim=48)
    pca.fit(X[:800])
    index = ADCIndex(pca.with_quantizer(bits=4)).add(X)
    idx = index.search(X[:5], k=3, rerank=5, originals=X)
    assert idx.shape == (5, 3)


def test_adc_index_empty_raises():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((600, 32)).astype(np.float32)
    pca = PCAMatryoshka(input_dim=32, output_dim=16)
    pca.fit(X)
    index = ADCIndex(pca.with_quantizer(bits=3))
    with pytest.raises(RuntimeError):
        index.search(X[:2], k=3)


def test_adc_hadamard_rotation():
    # ADCIndex must reproduce the exact reconstruct-cosine with a Hadamard-rotated
    # quantizer too (output_dim is a power of two, as Hadamard requires).
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 128)).astype(np.float32)
    X = (X @ (0.5 * rng.standard_normal((128, 128)) + np.eye(128))).astype(np.float32)
    train, DB = X[:2000], X[2000:]
    Q = rng.standard_normal((20, 128)).astype(np.float32)

    pca = PCAMatryoshka(input_dim=128, output_dim=64)
    pca.fit(train)
    pipe = pca.with_quantizer(bits=3, rotation="hadamard")
    index = ADCIndex(pipe).add(DB)

    q_rot, qbias = index._query_terms(Q)
    adc_idx, adc_sc = index._search_numpy(q_rot, qbias, len(DB))
    adc_full = np.take_along_axis(adc_sc, np.argsort(adc_idx, axis=1), axis=1)

    recon = pipe.decompress_batch(pipe.compress_batch(DB))
    qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    rn = recon / np.maximum(np.linalg.norm(recon, axis=1, keepdims=True), 1e-30)
    exact = qn @ rn.T

    assert np.abs(adc_full - exact).max() < 2e-4


@pytest.mark.parametrize("whiten", [False, True])
def test_adc_matches_reconstruct_cosine(whiten):
    # The ADC scorer must reproduce cos(q, decompress(compress(db))) exactly, for both
    # whiten settings. Regression test for the whiten bug: previously the DB projection
    # was whitened (via pca.transform) but the query terms and reconstruction norm were
    # not, so whiten=True silently mis-scored.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 64)).astype(np.float32)
    X = (X @ (0.5 * rng.standard_normal((64, 64)) + np.eye(64))).astype(np.float32)
    train, DB = X[:2000], X[2000:]
    Q = rng.standard_normal((20, 64)).astype(np.float32)

    pca = PCAMatryoshka(input_dim=64, output_dim=32, whiten=whiten)
    pca.fit(train)
    pipe = pca.with_quantizer(bits=3)
    index = ADCIndex(pipe).add(DB)

    q_rot, qbias = index._query_terms(Q)
    adc_idx, adc_sc = index._search_numpy(q_rot, qbias, len(DB))
    adc_full = np.take_along_axis(adc_sc, np.argsort(adc_idx, axis=1), axis=1)

    recon = pipe.decompress_batch(pipe.compress_batch(DB))
    qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    rn = recon / np.maximum(np.linalg.norm(recon, axis=1, keepdims=True), 1e-30)
    exact = qn @ rn.T

    assert np.abs(adc_full - exact).max() < 2e-4
