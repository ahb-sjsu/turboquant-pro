"""AATC reimplementation (benchmarks/kvquant_matrix/aatc.py), checked against its paper.

  the transform inverts exactly and its coefficients have second moments S;
  reverse water filling spends exactly the budget, protects layers 0-2 as the paper
      does, and equals a brute-force greedy allocation;
  the quantizer is the paper's token-scaled symmetric uniform quantizer;
  the value weight equals a brute-force output perturbation through W_O (GQA);
  calibration and the coding hooks run on a tiny random Qwen2: only tokens
      [sink, T - hot) change, the switch the KL pass uses turns them off, and stored
      bits are accounted;
  the harness's KIVI value grouping quantizes each group on its own.

Usage:
    python -m pytest tests/test_aatc.py -q
"""

from __future__ import annotations

import heapq
import importlib
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kvquant_matrix"))
sys.path.insert(0, str(ROOT / "tests"))


def _clear():
    for k in [k for k in os.environ if k.startswith("AATC") or k == "VAL_GROUP"]:
        os.environ.pop(k)


@pytest.fixture(autouse=True)
def _clean_env():
    """AATC's knobs are read at import from the environment: leave none behind, or
    every later harness import in the session runs as an AATC arm."""
    yield
    _clear()
    import aatc

    aatc.remove()
    importlib.reload(aatc)


def _ac(**env):
    _clear()
    os.environ.update({k: str(v) for k, v in env.items()})
    import aatc

    return importlib.reload(aatc)


def test_transform_inverts_and_whitens_to_s():
    ac = _ac()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4000, 2, 16, generator=g, dtype=torch.float64) @ torch.randn(
        16, 16, generator=g, dtype=torch.float64
    )
    m2 = torch.einsum("ngi,ngj->gij", x, x) / x.shape[0]
    b, binv, s = ac.fit_transform(m2)
    eye = torch.eye(16, dtype=torch.float64).expand(2, 16, 16)
    torch.testing.assert_close(binv @ b, eye, atol=1e-9, rtol=0)
    h = torch.einsum("ngi,gij->ngj", x, binv)
    got = torch.einsum("ngi,ngj->gij", h, h) / x.shape[0]
    torch.testing.assert_close(got, torch.diag_embed(s), atol=1e-8, rtol=1e-8)
    assert (s[:, :-1] >= s[:, 1:]).all()  # ordered by singular value


def _greedy(costs, budget, bmax):
    """Brute force: give one bit at a time to the largest marginal gain."""
    bits = [0] * len(costs)
    heap = [(-0.75 * c, i) for i, c in enumerate(costs)]
    heapq.heapify(heap)
    for _ in range(budget):
        g, i = heapq.heappop(heap)
        bits[i] += 1
        if bits[i] < bmax:
            heapq.heappush(heap, (g / 4, i))
    return bits


def _loss(costs, bits):
    return sum(c * 4.0 ** (-b) for c, b in zip(costs, bits))


@pytest.mark.parametrize("avg", [2.0, 3.0, 4.58])
def test_allocation_spends_the_budget_protects_and_is_optimal(avg):
    ac = _ac()
    g = torch.Generator().manual_seed(1)
    costs = [torch.rand(32, generator=g, dtype=torch.float64) ** 3 for _ in range(6)]
    bits = ac.allocate(costs, avg, bonus=1, bmax=16)
    total = sum(int(b.sum()) for b in bits)
    assert total == round(avg * 6 * 32)
    for li in range(3):  # protected: the average plus one, every channel
        assert (bits[li] == round(avg) + 1).all()
    free = torch.cat(costs[3:]).tolist()
    got = torch.cat(bits[3:]).tolist()
    want = _greedy(free, sum(got), 16)
    assert _loss(free, got) == pytest.approx(_loss(free, want), rel=1e-12)
    assert max(got) <= 16 and min(got) >= 0


def test_quantizer_is_the_papers():
    ac = _ac()
    h = torch.tensor([[1.0, -0.5, 0.25, 2.0]])
    s = torch.tensor([1.0, 1.0, 0.5, 4.0])
    bits = torch.tensor([3, 2, 0, 1])
    out = ac.quantize(h, bits, s)
    alpha = max(1.0, 0.5, 0.5, 0.5)  # max |h / s|
    for c in range(4):
        b = int(bits[c])
        if b == 0:
            assert out[0, c] == 0
            continue
        step = 2 * alpha * float(s[c]) / 2**b
        q = min(
            max(int(torch.floor(h[0, c] / step + 0.5)), -(2 ** (b - 1))),
            2 ** (b - 1) - 1,
        )
        assert float(out[0, c]) == pytest.approx(q * step)
    fine = ac.quantize(h, torch.full((4,), 16), s)
    assert (fine - h).abs().max() < 1e-3


def test_value_weight_equals_brute_force_output_perturbation():
    ac = _ac()
    g = torch.Generator().manual_seed(3)
    hq, hkv, d, groups, hidden = 4, 2, 4, 1, 6
    c = hkv * d // groups
    b = torch.randn(groups, c, c, generator=g, dtype=torch.float64)
    wo = torch.randn(hidden, hq * d, generator=g, dtype=torch.float64)
    w = ac.value_weight(b, wo, hq, hkv, d, groups)
    grp = hq // hkv
    want = torch.zeros(c, dtype=torch.float64)
    for ch in range(c):
        dv = b[0][
            ch
        ]  # a unit change of coefficient ch moves the group's values by B[ch]
        for r in range(
            hq
        ):  # query head r reads KV head r // grp, attending to this token
            kv = r // grp
            out = dv[kv * d : (kv + 1) * d] @ wo[:, r * d : (r + 1) * d].T
            want[ch] += float((out**2).sum())
    torch.testing.assert_close(w, want)


def _tiny(layers=5):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(
        vocab_size=300,
        hidden_size=512,
        intermediate_size=256,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=1024,
    )
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    return m


@pytest.mark.parametrize("transform", [1, 0])
def test_calibrate_install_code_and_account(transform, tmp_path):
    ac = _ac(
        AATC="kv",
        AATC_KEY_BITS=2,
        AATC_VAL_BITS=2,
        AATC_TRANSFORM=transform,
        AATC_CALIB=str(tmp_path / "cal.pt"),
    )
    model = _tiny()
    g = torch.Generator().manual_seed(4)
    wins = [torch.randint(0, 300, (1, 256), generator=g) for _ in range(6)]
    cal = ac.calibrate(model, wins[:3], wins[3:], ac.CALIB)
    assert len(cal["layers"]) == 5 and cal["meta"]["groups"] == 1
    for rec in cal["layers"]:
        for kind in ("k", "v"):
            assert (rec[kind]["s"] > 0).all() and (rec[kind]["sig2"] > 0).all()
    coder = ac.Coder(cal, 2.0, 2.0)
    for kind in ("k", "v"):
        total = sum(int(coder.bits[(li, kind)].sum()) for li in range(5))
        assert total == 2 * 5 * 512

    seen, on = [], [True]
    ac.install(
        model,
        cal,
        4,
        32,
        lambda li, n, s, cd: seen.append((li, n, s)),
        enabled=lambda: on[0],
    )
    x = torch.randint(0, 300, (1, 100), generator=g)
    k_out = {}

    def grab(mod, inp, out):
        k_out["k"] = out

    h = model.model.layers[0].self_attn.k_proj.register_forward_hook(grab)
    try:
        with torch.no_grad():
            model(x, use_cache=False)
        coded = k_out["k"]
        on[0] = False
        with torch.no_grad():
            model(x, use_cache=False)
        plain = k_out["k"]
    finally:
        h.remove()
        ac.remove()
    diff = (coded - plain).abs().amax(-1)[0]
    assert diff[:4].max() == 0 and diff[68:].max() == 0  # sink and hot window untouched
    assert (diff[4:68] > 0).all()
    assert seen and all(n == 64 and s == 4 for _, n, s in seen)
    a = ac.account(64, 4, 4, 128, coder, 4)
    assert a["code"] == 64 * coder.code_bits(4, "k") and a["meta"] == 64 * 16
    assert a["outliers"] == 4 * 512 * 16 and a["elems"] == 68 * 512


def test_kivi_value_groups_are_quantized_separately():
    from test_key_coding import SHIPPED, _harness

    h = _harness(
        **{
            **SHIPPED,
            "CODEBOOK": "uniform",
            "KEY_BITS": 2,
            "VAL_BITS": 2,
            "VAL_GROUP": 32,
        }
    )
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 2, 10, 128, generator=g).half()
    x[..., :32] *= 100  # one loud group must not coarsen the others
    y = h.qdq_val_block(x)
    for k in range(4):  # each group alone through the whole-head path: the same bits
        part = x[..., 32 * k : 32 * (k + 1)]
        torch.testing.assert_close(
            y[..., 32 * k : 32 * (k + 1)], h._qdq_val(part), rtol=0, atol=0
        )
    os.environ.pop("VAL_GROUP")
