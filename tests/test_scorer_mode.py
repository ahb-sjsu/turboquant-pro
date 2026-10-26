# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""The scorer is the caller's choice and every search names it.

``mode="exact"`` is the float reference, ``mode="fast"`` the compiled kernel
where it can run; storage and deployment (memory-mapping, ``block``, whether a
kernel is built) never choose silently. After a search, ``last_scorer`` says
which scorer ran, what it promises, and why ``"fast"`` fell back when it did
(docs/DESIGN_fast_adc.md, "The contract, in one place").
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant_pro import IVFIndex, TQEIndex, _adc, telemetry
from turboquant_pro import adc_index as ai
from turboquant_pro import scorer as S

kernel_only = pytest.mark.skipif(
    not _adc.is_available(), reason="needs the compiled ADC kernel"
)


def _corpus(n=800, dim=48, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


@pytest.fixture
def no_kernel(monkeypatch):
    monkeypatch.setattr(ai._adc, "load", lambda: None)
    return True


# --------------------------------------------------------------------------- #
# The mode argument                                                            #
# --------------------------------------------------------------------------- #


def test_mode_defaults_to_fast_and_exact_true_is_its_older_spelling():
    assert S.resolve_mode(None, False) == S.FAST
    assert S.resolve_mode(None, True) == S.EXACT
    assert S.resolve_mode("exact", True) == S.EXACT
    with pytest.raises(ValueError, match="contradicts"):
        S.resolve_mode("fast", True)
    with pytest.raises(ValueError, match="mode must be one of"):
        S.resolve_mode("quick", False)


def test_exact_mode_and_exact_true_return_the_same_ranking():
    x = _corpus()
    idx = TQEIndex.create(x, output_dim=32, bits=4, seed=1)
    a, asc = idx.search(x[:20], k=10, mode="exact")
    b, bsc = idx.search(x[:20], k=10, exact=True)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(asc, bsc)


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #


def test_exact_mode_names_the_reference_and_no_fallback():
    x = _corpus()
    idx = TQEIndex.create(x, output_dim=32, bits=4, seed=1)
    assert idx.last_scorer is None
    idx.search(x[:5], k=10, mode="exact")
    p = idx.last_scorer
    assert p["mode"] == "exact"
    assert p["first_stage"] == {"scorer": "exact-float", "semantics": "reference"}
    assert p["fallback_reason"] is None and p["rerank"] is None


def test_fast_without_a_kernel_runs_the_reference_and_says_why(no_kernel):
    x = _corpus()
    idx = TQEIndex.create(x, output_dim=32, bits=4, seed=1)
    idx.search(x[:5], k=10)
    p = idx.last_scorer
    assert p["mode"] == "fast" and p["first_stage"]["scorer"] == "exact-float"
    assert p["fallback_reason"] == "no compiled kernel is built"
    assert S.describe()["fast"] == "exact-float" and S.describe()["kernel"] is None


def test_fast_on_a_memory_mapped_index_says_it_ran_the_reference(tmp_path):
    x = _corpus()
    p = tmp_path / "i.tqix"
    TQEIndex.create(x, output_dim=32, bits=4, seed=1).save(str(p))
    mm = TQEIndex.open(str(p), mmap=True)
    mm.search(x[:5], k=10)
    prov = mm.last_scorer
    assert prov["first_stage"]["scorer"] == "exact-float"
    assert prov["fallback_reason"] in (
        "no compiled kernel is built",
        "memory-mapped index: the kernel scans RAM only",
    )


def test_fast_with_a_block_says_it_ran_the_reference():
    x = _corpus()
    idx = TQEIndex.create(x, output_dim=32, bits=4, seed=1)
    idx.search(x[:5], k=10, block=128)
    assert idx.last_scorer["first_stage"]["scorer"] == "exact-float"
    assert idx.last_scorer["fallback_reason"] is not None


def test_the_rerank_is_recorded_with_its_width_and_basis():
    x = _corpus()
    with_orig = TQEIndex.create(x, output_dim=32, bits=4, seed=1)
    with_orig.search(x[:5], k=10, rerank=4, mode="exact")
    assert with_orig.last_scorer["rerank"] == {
        "width": 40,
        "basis": "originals",
        "semantics": "exact",
    }
    codes_only = TQEIndex.create(x, output_dim=32, bits=4, seed=1, keep_originals=False)
    codes_only.search(x[:5], k=10, rerank=4, mode="exact")
    assert codes_only.last_scorer["rerank"]["basis"] == "reconstruction"
    assert codes_only.last_scorer["rerank"]["semantics"] == "approximate"


def test_scorer_answers_before_searching_what_the_search_records():
    x = _corpus()
    idx = TQEIndex.create(x, output_dim=32, bits=4, seed=1)
    for mode in ("exact", "fast"):
        before = idx.scorer(mode)
        idx.search(x[:3], k=5, mode=mode)
        assert before["first_stage"] == idx.last_scorer["first_stage"]
        assert before["fallback_reason"] == idx.last_scorer["fallback_reason"]


def test_a_trace_carries_the_scorer():
    x = _corpus()
    idx = TQEIndex.create(x, output_dim=32, bits=4, seed=1)
    with telemetry.scope(telemetry.Tracer()) as sc:
        idx.search(x[:2], k=5, mode="exact")
    assert sc.last["scorer"] == idx.last_scorer
    assert sc.last["params"]["mode"] == "exact"


def test_the_flat_and_ivf_indexes_take_the_mode_too():
    x = _corpus(1200, 48)
    idx = TQEIndex.create(x, output_dim=32, bits=4, seed=1)
    idx._adc.search(x[:3], k=5, mode="exact")
    assert idx._adc.last_scorer["first_stage"]["scorer"] == "exact-float"
    ivf = IVFIndex.create(x, output_dim=32, bits=4)
    ivf.search(x[:3], k=5, nprobe=4, mode="exact")
    assert ivf.last_scorer["first_stage"]["scorer"] == "exact-float"
    assert ivf.last_scorer["fallback_reason"] is None


def test_the_certificate_environment_names_the_scorers():
    from turboquant_pro.certify_report import _certify_environment

    env = _certify_environment()["scorers"]
    assert env["reference"] == "exact-float"
    assert env["fast"] in ("exact-float", "kernel-uint8-lut", "kernel-float-lut")


# --------------------------------------------------------------------------- #
# With the kernel built                                                        #
# --------------------------------------------------------------------------- #


@kernel_only
def test_fast_with_the_kernel_names_an_approximate_scorer_and_its_table():
    x = _corpus(1500, 64)
    idx = TQEIndex.create(x, output_dim=32, bits=4, seed=3)
    idx.search(x[:5], k=10)
    first = idx.last_scorer["first_stage"]
    assert first["scorer"] in ("kernel-uint8-lut", "kernel-float-lut")
    assert first["semantics"] == "approximate"
    assert first["kernel"]["lut_levels"] == 255
    assert first["kernel"]["lut_scale"] == "query-global"
    assert idx.last_scorer["fallback_reason"] is None


@kernel_only
def test_the_ivf_fast_scan_names_the_kernel():
    x = _corpus(1500, 64)
    ivf = IVFIndex.create(x, output_dim=32, bits=4)
    ivf.search(x[:3], k=5, nprobe=4)
    assert ivf.last_scorer["first_stage"]["semantics"] == "approximate"
