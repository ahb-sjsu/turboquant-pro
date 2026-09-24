"""Key-coding stages of the KV harness (observer-advantage Part II), synthetic tensors.

The stages sit around the shipped key codebook
(benchmarks/kvquant_matrix/key_coding.py).
These tests pin what the registration relies on before any model runs:

  G0  every basis, with the codebook replaced by the identity, returns the native keys
      (so a basis changes only what the codebook sees);
  the batched observer maps are Part I's maps (observer_advantage.cell.balanced_o and
      consumer_basis.run.bases), compared through sign-invariant products;
  water-filling spends exactly D * KEY_BITS bits per head and is optimal;
  byte matching buys outliers worth exactly one dense basis;
  the shipped arm is unchanged when every stage is at its default;
  configurations the stages cannot honour are refused.

Usage:
    python -m pytest tests/test_key_coding.py -q
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import os
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
KVDIR = ROOT / "benchmarks" / "kvquant_matrix"
sys.path.insert(0, str(KVDIR))
sys.path.insert(0, str(ROOT / "benchmarks"))

STAGE_VARS = (
    "KEY_BASIS",
    "BASIS_FIT",
    "KEY_ALLOC",
    "BYTE_MATCH",
    "BASIS_SEED",
    "CODEBOOK",
    "KEY_BITS",
    "OUTLIER_FRAC",
    "SINK",
    "PREROPE",
    "NOQUANT",
    "ALLOC_BMIN",
    "ALLOC_BMAX",
    "BASIS_CALIB",
    "KEY_JITTER",
)


def _kc(**env):
    """key_coding re-imported under ``env`` (its knobs are read at import)."""
    for k in STAGE_VARS:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    import key_coding

    return importlib.reload(key_coding)


def _harness(**env):
    pytest.importorskip("transformers")
    _kc(**env)
    os.environ.setdefault("SHARD_ID", "0")
    os.environ.setdefault("NUM_SHARDS", "1")
    os.environ["OUT_DIR"] = tempfile.mkdtemp(prefix="tq_keycoding_")
    spec = importlib.util.spec_from_file_location(
        "tq_paper_lb_shard", KVDIR / "tq_paper_lb_shard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spd(d, gen, scale=1.0):
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    return scale * (a @ a.T / d + 0.05 * torch.eye(d, dtype=torch.float64))


def _keys_queries(b=1, hkv=2, g=2, n=96, t=128, d=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mix = torch.randn(d, d, generator=gen) / d**0.5
    k = torch.randn(b, hkv, n, d, generator=gen) @ mix
    # A DC-offset channel, the structure the shipped codebook relies on.
    k[..., 3] += 4.0
    q = torch.randn(b, hkv * g, t, d, generator=gen)
    q[..., 5] *= 6.0  # a dominant read channel
    return k, q


# ------------------------------------------------------------------ #
# Part I's maps                                                      #
# ------------------------------------------------------------------ #


def test_eig_floor_matches_consumer_basis():
    arms = pytest.importorskip("consumer_basis.arms")
    assert _kc().EIG_FLOOR == arms.EIG_FLOOR


@pytest.mark.parametrize("balanced", [True, False])
def test_observer_maps_equal_part_one(balanced):
    run = pytest.importorskip("consumer_basis.run")
    kc = _kc()
    d = 12
    gen = torch.Generator().manual_seed(1)
    s, c = _spd(d, gen), _spd(d, gen, 3.0)
    a, b = kc.observer_maps(s.unsqueeze(0), c.unsqueeze(0), balanced=balanced)
    if balanced:
        cell = pytest.importorskip("observer_advantage.cell")
        qa, db = cell.balanced_o(s.numpy(), c.numpy(), d)
    else:
        qa, db = run.bases(s.numpy(), c.numpy(), d)["O"]
    a, b = a[0].numpy(), b[0].numpy()
    # Singular vectors are defined up to a joint sign per pair; these products are not.
    for x, y in ((a.T @ b, qa.T @ db), (a.T @ a, qa.T @ qa), (b.T @ b, db.T @ db)):
        np.testing.assert_allclose(x, y, rtol=1e-7, atol=1e-9)


def test_balanced_split_preserves_every_logit():
    kc = _kc()
    gen = torch.Generator().manual_seed(2)
    s, c = _spd(10, gen).unsqueeze(0), _spd(10, gen).unsqueeze(0)
    a, b = kc.observer_maps(s, c)
    np.testing.assert_allclose(
        (a.transpose(-1, -2) @ b)[0].numpy(), np.eye(10), atol=1e-8
    )


# ------------------------------------------------------------------ #
# G0: the injection is exact                                         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("basis", ["native", "P", "O", "Oey", "O_foreign", "R", "H"])
def test_g0_identity_codebook_returns_native_keys(basis):
    kc = _kc(KEY_BASIS=basis)
    k, q = _keys_queries()
    k = k.half()
    out = kc.code_keys(k, q, layer=3, quantize=lambda z, bits: z, key_bits=4)
    assert torch.equal(out, k)  # bit for bit: the residual form


@pytest.mark.parametrize("basis", ["P", "O", "R", "O_foreign"])
def test_quantization_error_is_carried_back_through_the_inverse(basis):
    kc = _kc(KEY_BASIS=basis)
    k, q = _keys_queries()
    s, c = kc.key_moment(k), kc.query_moment(q, k.shape[1])
    bmap = kc.key_map(basis, s, c, 3, k.shape[1], k.shape[3])
    delta = 1e-3 * torch.arange(k.shape[3], dtype=torch.float32)  # per coded coordinate
    out = kc.code_keys(k, q, layer=3, quantize=lambda z, bits: z + delta, key_bits=4)
    want = (
        k.double()
        + torch.einsum("hij,j->hi", torch.linalg.inv(bmap), delta.double())[
            None, :, None, :
        ]
    )
    torch.testing.assert_close(out.double(), want, rtol=1e-5, atol=1e-6)


def test_jitter_moves_every_key_by_exactly_one_ulp():
    kc = _kc(KEY_JITTER=1, BASIS_SEED=3)
    k = _keys_queries()[0].half()
    j = kc.jitter(k, layer=5)
    steps = (j.view(torch.int16).int() - k.view(torch.int16).int()).abs()
    assert (steps == 1).all()
    assert torch.equal(j, kc.jitter(k, layer=5))  # seeded
    assert (j.float() - k.float()).abs().max() < 0.01


def test_jitter_never_makes_a_nan_or_inf():
    kc = _kc(KEY_JITTER=1)
    edge = torch.tensor(
        [0.0, -0.0, 65504.0, -65504.0, 6e-8, -6e-8, 1.0], dtype=torch.float16
    )
    k = edge.repeat(64).reshape(1, 1, 64, 7)
    j = kc.jitter(k, layer=0)
    assert torch.isfinite(j).all()
    assert (j != k).all()  # every element moved
    assert ((j.float() - k.float()).abs() <= 32.0).all()  # one ulp even at the top


@pytest.mark.parametrize("basis", ["P", "O", "R"])
def test_logits_through_the_query_map(basis):
    kc = _kc(KEY_BASIS=basis)
    k, q = _keys_queries()
    s, c = kc.key_moment(k), kc.query_moment(q, k.shape[1])
    bmap = kc.key_map(basis, s, c, 0, k.shape[1], k.shape[3])
    amap = torch.linalg.inv(bmap).transpose(-1, -2)
    z = torch.einsum("hij,bhnj->bhni", bmap, k.double())
    qg = q.double().reshape(1, 2, 2, -1, k.shape[3])  # GQA readers of each KV head
    lhs = torch.einsum("hij,bhgtj,bhni->bhgtn", amap, qg, z)
    rhs = torch.einsum("bhgtd,bhnd->bhgtn", qg, k.double())
    torch.testing.assert_close(lhs, rhs, rtol=1e-8, atol=1e-8)


def test_query_moment_pools_each_kv_heads_readers():
    kc = _kc()
    q = torch.zeros(1, 4, 5, 3)
    q[:, 2:, :, 0] = 1.0  # only KV head 1's readers carry energy
    c = kc.query_moment(q, 2)
    assert c[0].abs().sum() == 0 and c[1][0, 0] == 1.0


def test_foreign_observer_reads_the_next_group():
    kc = _kc(KEY_BASIS="O_foreign")
    k, q = _keys_queries(hkv=3)
    s, c = kc.key_moment(k), kc.query_moment(q, 3)
    foreign = kc.key_map("O_foreign", s, c, 0, 3, k.shape[3])
    own_next = kc.observer_maps(s, c.roll(-1, dims=0))[1]
    torch.testing.assert_close(foreign, own_next)


@pytest.mark.parametrize("kind", ["R", "H"])
def test_seeded_bases_are_orthogonal_and_reproducible(kind):
    kc = _kc(BASIS_SEED=7)
    b1 = kc.seeded_basis(kind, 3, 16, layer=2)
    b2 = kc.seeded_basis(kind, 3, 16, layer=2)
    torch.testing.assert_close(b1, b2)
    eye = torch.eye(16, dtype=torch.float64).expand(3, 16, 16)
    torch.testing.assert_close(b1 @ b1.transpose(-1, -2), eye, rtol=0, atol=1e-10)
    assert not torch.allclose(b1, kc.seeded_basis(kind, 3, 16, layer=3))


# ------------------------------------------------------------------ #
# Allocation and bytes                                               #
# ------------------------------------------------------------------ #


def test_water_fill_spends_the_budget_within_bounds():
    kc = _kc()
    cost = torch.rand(4, 32, generator=torch.Generator().manual_seed(3)).double() ** 4
    bits = kc.water_fill(cost, 3)
    assert (bits.sum(1) == 32 * 3).all()
    assert bits.min() >= kc.ALLOC_BMIN and bits.max() <= kc.ALLOC_BMAX


def test_water_fill_is_optimal_on_a_small_instance():
    kc = _kc(ALLOC_BMIN=1, ALLOC_BMAX=5)
    cost = torch.tensor([[9.0, 1.0, 0.2, 4.0]], dtype=torch.float64)
    got = kc.water_fill(cost, 3)[0]

    def err(b):
        return float((cost[0] * 0.25 ** torch.tensor(b, dtype=torch.float64)).sum())

    best = min(
        (b for b in itertools.product(range(1, 6), repeat=4) if sum(b) == 12), key=err
    )
    assert err(got.tolist()) == pytest.approx(err(best))


def test_read_allocation_follows_the_reader_in_native_channels():
    kc = _kc(KEY_ALLOC="read")
    k, q = _keys_queries()
    bits = kc.allocation(
        k.double(),
        torch.eye(16, dtype=torch.float64).expand(2, 16, 16),
        kc.query_moment(q, 2),
        4,
    )
    assert (bits[:, 5] >= bits.median(dim=1).values).all()  # the dominant read channel


def test_byte_match_buys_exactly_one_dense_basis():
    kc = _kc(BYTE_MATCH=1)
    n, d, kb = 3000, 128, 4
    frac = kc.extra_outlier_frac(n, d, kb)
    assert frac * n * d * (kc.OUTLIER_BITS - kb) == pytest.approx(d * d * 16)
    assert _kc().extra_outlier_frac(n, d, kb) == 0.0


def test_account_charges_the_dense_basis_only_to_dense_bases():
    for basis, charged in (("O", True), ("P", True), ("R", False), ("native", False)):
        a = _kc(KEY_BASIS=basis).account(256, 2, 16, 4, 32, 0.0, 0, 2)
        assert (a["basis"] == 2 * 16 * 16 * 16) is charged
        assert a["total"] == sum(
            a[x] for x in ("code", "meta", "outliers", "basis", "alloc")
        )


# ------------------------------------------------------------------ #
# Inside the harness                                                 #
# ------------------------------------------------------------------ #

SHIPPED = dict(CODEBOOK="nf4a", KEY_BITS=4, OUTLIER_FRAC=0.02, SINK=4)


def test_shipped_arm_is_unchanged_with_default_stages():
    h = _harness(**SHIPPED)
    k, _ = _keys_queries(n=160)
    x = k.half()
    manual = h._quant_nf4a_group(x, h.G, h.NF4)
    keep = torch.zeros_like(x, dtype=torch.bool)
    keep[:, :, :4, :] = True
    kk = max(1, int(round(x.shape[2] * 0.02)))
    thr = x.abs().kthvalue(x.shape[2] - kk + 1, dim=2, keepdim=True).values
    keep |= x.abs() >= thr
    torch.testing.assert_close(h._code_settled_keys(x, 0), torch.where(keep, x, manual))


@pytest.mark.parametrize(
    "basis,alloc,codebook",
    [
        ("O", "uniform", "nf4a"),
        ("P", "uniform", "nf4a"),
        ("R", "uniform", "nf4a"),
        ("native", "read", "uniform"),
        ("O", "read", "uniform"),
        ("native", "key", "kivi"),
    ],
)
def test_staged_arms_run_in_the_harness(basis, alloc, codebook):
    h = _harness(
        **{**SHIPPED, "CODEBOOK": codebook, "KEY_BASIS": basis, "KEY_ALLOC": alloc}
    )
    k, q = _keys_queries(n=160, t=200)
    h._LAST_Q = q.half()
    h._ACCT.clear()
    out = h._code_settled_keys(k.half(), 0)
    assert out.shape == k.shape and torch.isfinite(out).all()
    err = (out.float() - k).norm() / k.norm()
    assert 0 < err < 0.5  # quantized, and not destroyed
    assert h._bits_summary()["total"] > 4


def test_byte_matched_native_keeps_more_fp16_entries():
    base = _harness(**SHIPPED)
    k, _ = _keys_queries(n=160)
    same = (base._code_settled_keys(k.half(), 0) == k.half()).sum()
    bm = _harness(**SHIPPED, BYTE_MATCH=1)
    more = (bm._code_settled_keys(k.half(), 0) == k.half()).sum()
    assert more > same


@pytest.mark.parametrize(
    "env",
    [
        dict(CODEBOOK="nf4a", KEY_BITS=4, KEY_ALLOC="read"),
        dict(CODEBOOK="uniform", KEY_BASIS="O", PREROPE=1),
        dict(KEY_BASIS="Q"),
        dict(CODEBOOK="uniform", KEY_BASIS="R", BASIS_FIT="calib"),
    ],
)
def test_unhonourable_configurations_are_refused(env):
    kc = _kc(**env)
    with pytest.raises(SystemExit):
        kc.validate(
            os.environ.get("CODEBOOK", "uniform"),
            int(os.environ.get("PREROPE", "0")),
            0,
        )


def test_resume_keeps_whole_lines_and_cuts_a_torn_one(tmp_path):
    h = _harness()
    p = tmp_path / "qasper.0.jsonl"
    p.write_text('{"idx": 0, "pred": "a"}\n{"idx": 4, "pred": "b"}\n{"idx": 8, "pr')
    assert h._resume_done(str(p)) == {0, 4}
    assert p.read_text().count("\n") == 2 and p.read_text().endswith("}\n")
    assert h._resume_done(str(tmp_path / "missing.jsonl")) == set()
