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
