"""Tests for FAISS integration."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import faiss  # noqa: F401

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from turboquant_pro.pca import PCAMatryoshka

pytestmark = pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")


@pytest.fixture()
def pca_fitted():
    """Return a PCA fitted on random data."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((500, 128)).astype(np.float32)
    pca = PCAMatryoshka(input_dim=128, output_dim=64)
    pca.fit(data)
    return pca, data


class TestTurboQuantFAISS:
    """Tests for the FAISS wrapper."""

    def test_create_flat_index(self, pca_fitted) -> None:
        from turboquant_pro.faiss_index import TurboQuantFAISS

        pca, _ = pca_fitted
        tqf = TurboQuantFAISS(pca, index_type="flat")
        assert tqf.n_vectors == 0
        assert tqf.compression_ratio == 2.0  # 128 -> 64

    def test_add_and_search(self, pca_fitted) -> None:
        from turboquant_pro.faiss_index import TurboQuantFAISS

        pca, data = pca_fitted
        tqf = TurboQuantFAISS(pca, index_type="flat", metric="ip")
        tqf.add(data)
        assert tqf.n_vectors == 500

        # Search for first vector — should find itself
        distances, indices = tqf.search(data[0], k=5)
        assert indices.shape == (1, 5)
        assert indices[0, 0] == 0  # Self is closest

    def test_batch_search(self, pca_fitted) -> None:
        from turboquant_pro.faiss_index import TurboQuantFAISS

        pca, data = pca_fitted
        tqf = TurboQuantFAISS(pca, index_type="flat")
        tqf.add(data)

        queries = data[:10]
        distances, indices = tqf.search(queries, k=3)
        assert distances.shape == (10, 3)
        assert indices.shape == (10, 3)

    def test_ivf_index(self, pca_fitted) -> None:
        from turboquant_pro.faiss_index import TurboQuantFAISS

        pca, data = pca_fitted
        tqf = TurboQuantFAISS(pca, index_type="ivf", n_lists=10, n_probe=5)
        tqf.add(data)
        assert tqf.n_vectors == 500

        distances, indices = tqf.search(data[0], k=5)
        assert indices.shape == (1, 5)

    def test_stats(self, pca_fitted) -> None:
        from turboquant_pro.faiss_index import TurboQuantFAISS

        pca, data = pca_fitted
        tqf = TurboQuantFAISS(pca, index_type="flat")
        tqf.add(data)

        s = tqf.stats()
        assert s["n_vectors"] == 500
        assert s["input_dim"] == 128
        assert s["index_dim"] == 64
        assert s["dim_compression"] == 2.0
        assert s["saved_mb"] > 0

    def test_invalid_index_type(self, pca_fitted) -> None:
        from turboquant_pro.faiss_index import TurboQuantFAISS

        pca, _ = pca_fitted
        with pytest.raises(ValueError, match="Unknown index_type"):
            TurboQuantFAISS(pca, index_type="invalid")
