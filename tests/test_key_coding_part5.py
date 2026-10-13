"""Part V key allocations (docs/PREREG_full_consumer_kv.md), synthetic tensors.

Pins what the registration relies on before any model runs:

  the head-output weight equals a brute-force finite difference of the heads' outputs
      through the softmax, the values and W_O (MHA and GQA, any query block size);
  head and fisher allocate in native channels, spend exactly D * KEY_BITS bits per head
      and pass G0 (the identity codebook returns the native keys);
  calibrated weights round-trip through the calibration file;
  configurations the stages cannot honour are refused;
  the harness codes settled keys with the head weight from the full prefill.

Usage:
    python -m pytest tests/test_key_coding_part5.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kvquant_matrix"))
sys.path.insert(0, str(ROOT / "tests"))

from test_key_coding import SHIPPED, _harness, _kc  # noqa: E402


def _head(b=1, hkv=2, grp=2, t=11, d=8, hidden=12, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, hkv * grp, t, d, generator=gen, dtype=torch.float64)
    k = torch.randn(b, hkv, t, d, generator=gen, dtype=torch.float64)
    v = torch.randn(b, hkv, t, d, generator=gen, dtype=torch.float64)
    wo = torch.randn(hidden, hkv * grp * d, generator=gen, dtype=torch.float64)
    return q, k, v, wo


def _outputs(q, k, v, wo):
    """(B, Hq, T, hidden): each query head's output through its slice of W_O."""
    b, hq, t, d = q.shape
    grp = hq // k.shape[1]
    out = []
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
    for g in range(hq):
        h = g // grp
        logit = (q[:, g] @ k[:, h].transpose(1, 2)) / math.sqrt(d)
        a = torch.softmax(logit.masked_fill(mask, float("-inf")), -1)
        out.append((a @ v[:, h]) @ wo[:, g * d : (g + 1) * d].T)
    return torch.stack(out, 1)


def _brute_force(q, k, v, wo, n, eps=1e-5):
    """D times the summed output energy per unit squared perturbation of each settled
    key element, summed over the settled keys: the definition head_weight must equal."""
    hkv, d = k.shape[1], k.shape[3]
    base = _outputs(q, k, v, wo)
    w = torch.zeros(hkv, d, dtype=torch.float64)
    for h in range(hkv):
        for t in range(n):
            for j in range(d):
                kp = k.clone()
                kp[:, h, t, j] += eps
                delta = _outputs(q, kp, v, wo) - base
                w[h, j] += d * float((delta**2).sum()) / eps**2
    return w


@pytest.mark.parametrize("grp,chunk", [(1, 512), (2, 512), (2, 3)])
def test_head_weight_equals_finite_difference(grp, chunk):
    kc = _kc(HEAD_CHUNK=chunk)
    q, k, v, wo = _head(grp=grp, hkv=2)
    n = 8
    got = kc.head_weight(q, k, v, wo, n)
    want = _brute_force(q, k, v, wo, n)
    torch.testing.assert_close(got, want, rtol=2e-3, atol=1e-6)


def test_head_weight_ignores_keys_with_equal_values():
    """Two keys with identical values: moving attention between them leaves every output
    unchanged, so channels read only through that exchange get no weight."""
    kc = _kc()
    q, k, v, wo = _head(hkv=1, grp=1, t=6)
    v[:] = v[:, :, :1]  # every value equal: the head output cannot change at all
    w = kc.head_weight(q, k, v, wo, 6)
    assert float(w.abs().max()) < 1e-8


@pytest.mark.parametrize(
    "alloc,fit", [("head", "prefill"), ("fisher", "calib"), ("head", "calib")]
)
def test_part5_allocations_spend_the_budget_and_pass_g0(alloc, fit, tmp_path):
    env = {
        **SHIPPED,
        "CODEBOOK": "uniform",
        "KEY_BITS": 3,
        "KEY_ALLOC": alloc,
        "BASIS_FIT": fit,
    }
    if fit == "calib":
        env["BASIS_CALIB"] = str(tmp_path / "cal.pt")
    kc = _kc(**env)
    q, k, v, wo = _head(hkv=2, grp=2, t=40, d=8)
    n = 32
    if fit == "calib":
        for li in (0,):
            kc.accumulate_weight(li, torch.rand(2, 8, dtype=torch.float64) + 0.1, 40)
        kc.save_calibration(env["BASIS_CALIB"])
    else:
        kc._WO[0] = wo.float()
    seen = {}

    def identity(z, bits):
        seen["bits"] = bits
        return z

    out = kc.code_keys(
        k[:, :, :n].float(), q.float(), 0, identity, 3, (k.float(), v.float())
    )
    torch.testing.assert_close(
        out, k[:, :, :n].float()
    )  # G0: identity codebook is exact
    bits = seen["bits"]
    assert bits.shape == (2, 8)
    assert bits.sum(1).tolist() == [8 * 3, 8 * 3]
    assert int(bits.min()) >= kc.ALLOC_BMIN and int(bits.max()) <= kc.ALLOC_BMAX


def test_head_allocation_favours_the_channel_the_output_reads():
    kc = _kc(**{**SHIPPED, "CODEBOOK": "uniform", "KEY_BITS": 3, "KEY_ALLOC": "head"})
    q, k, v, wo = _head(hkv=1, grp=1, t=40, d=8)
    q[..., 2] *= 8.0  # channel 2 dominates the logits, and so the output's sensitivity
    w = kc.head_weight(q, k, v, wo, 32)
    assert int(w[0].argmax()) == 2
    bits = kc.allocation(k[:, :, :32], None, w, 3)
    assert int(bits[0, 2]) == int(bits[0].max())


def test_calibrated_weight_round_trips(tmp_path):
    p = tmp_path / "cal.pt"
    kc = _kc(KEY_ALLOC="fisher", BASIS_FIT="calib", BASIS_CALIB=str(p))
    a, b = torch.rand(3, 4, dtype=torch.float64), torch.rand(3, 4, dtype=torch.float64)
    kc.accumulate_weight(5, a, 10)
    kc.accumulate_weight(5, b, 30)
    kc.save_calibration(str(p))
    torch.testing.assert_close(kc.calibration(5)[0], (a + b) / 40)


@pytest.mark.parametrize(
    "env",
    [
        {"KEY_ALLOC": "head", "KEY_BASIS": "O"},
        {"KEY_ALLOC": "fisher", "BASIS_FIT": "prefill"},
        {"KEY_ALLOC": "fisher", "KEY_BASIS": "P", "BASIS_FIT": "calib"},
    ],
)
def test_part5_unhonourable_configurations_are_refused(env):
    kc = _kc(**{**SHIPPED, "CODEBOOK": "uniform", "KEY_BITS": 3, **env})
    with pytest.raises(SystemExit):
        kc.validate("uniform", 0, 0)


def test_part5_calibration_windows_are_equal_and_longer_than_hot():
    """head_cal and fisher_cal fit on the same windows, each past the hot window (the
    first NRP smoke failed on Part II's short calibration passages)."""
    h = _harness(
        **{
            **SHIPPED,
            "CODEBOOK": "uniform",
            "KEY_BITS": 3,
            "KEY_ALLOC": "head",
            "BASIS_FIT": "calib",
        }
    )
    pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        wins = h._part5_windows(tok)
    except (OSError, ValueError) as e:  # no cached tokenizer or text offline
        pytest.skip(f"tokenizer or WikiText not available: {e!r}")
    assert len(wins) == h.KC.PART5_CALIB_N
    assert all(w.shape == (1, h.KC.PART5_CALIB_SEQ) for w in wins)
    assert h.KC.PART5_CALIB_SEQ > h.HOT
    assert len({tuple(w[0, :8].tolist()) for w in wins}) == len(wins)  # consecutive


def test_head_allocation_runs_in_the_harness():
    h = _harness(
        **{**SHIPPED, "CODEBOOK": "uniform", "KEY_BITS": 3, "KEY_ALLOC": "head"}
    )
    q, k, v, wo = _head(hkv=2, grp=2, t=200, d=16, hidden=24)
    h.KC._WO[0] = wo.half()
    h._LAST_Q = q.half()
    h._ACCT.clear()
    n = 160
    out = h._code_settled_keys(k[:, :, :n].half(), 0, (k.half(), v.half()))
    assert out.shape == (1, 2, n, 16) and torch.isfinite(out).all()
    err = (out.float() - k[:, :, :n].float()).norm() / k[:, :, :n].float().norm()
    assert 0 < err < 0.5
    assert h._bits_summary()["alloc"] > 0
