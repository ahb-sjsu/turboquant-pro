"""Expected Attention for Part V (benchmarks/kvquant_matrix/ea.py), against kvpress.

  the four vendored kvpress 0.5.5 files are the wheel's, byte for byte, and importing ea
      with EA unset loads none of them;
  matched bytes: u3's stored bits per element and the eviction ratio that gives fp16
      entries the same bytes, and the realized bits within the scorer's 1%;
  the hook keeps exactly what kvpress's ScorerPress.compress keeps, evicts at prefill,
      and the harness generates from the evicted cache with u3's coding off (ea) or on;
  perplexity: pass 2 with every entry kept is the plain forward, and a token evicted
      from every layer changes the logits from its position on and none before it;
  the runner's record and the scorer's check: ea held to u3's bytes, ea_u3 reported;
  every existing arm's code is unchanged since the registered 2e8956a (additions only,
      each gated on EA), and every existing arm's env line is the same.

Usage:
    python -m pytest tests/test_part5_ea.py -q
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
KVDIR = ROOT / "benchmarks" / "kvquant_matrix"
sys.path.insert(0, str(KVDIR))
sys.path.insert(0, str(ROOT / "tests"))

VENDORED = {
    "utils.py": "e8a19f23bfec0e06c058a9563b86f823935849ac7b93287657d602e4f694fce2",
    "presses/base_press.py": (
        "b37e5d08bce40ef90cd3dde1e9853ed7fa3fdc9996d42942a2dc6da0fd16fe1a"
    ),
    "presses/scorer_press.py": (
        "39191c9fb161238a2c44b32df8852e7406c3026423cd198470a0be026b8dfc50"
    ),
    "presses/expected_attention_press.py": (
        "6fbae9ab7b976e295959a53b80d145fac3687138cd91fa8fcf0e39d48f6f6c00"
    ),
}
REGISTERED = "2e8956abe483a0c43d1ff0a1cedad25d5a50ab3b"


def _clear():
    for k in [k for k in os.environ if k.startswith("EA")]:
        os.environ.pop(k)


@pytest.fixture(autouse=True)
def _clean_env():
    yield
    _clear()
    import ea

    ea.remove()
    importlib.reload(ea)


def _ea(**env):
    _clear()
    os.environ.update({k: str(v) for k, v in env.items()})
    import ea

    return importlib.reload(ea)


def _tiny(layers=3, family="qwen2"):
    transformers = pytest.importorskip("transformers")
    common = dict(
        vocab_size=300,
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=4096,
    )
    cfg = (
        transformers.Qwen2Config(**common)
        if family == "qwen2"
        else transformers.LlamaConfig(**common)
    )
    torch.manual_seed(0)
    cls = (
        transformers.Qwen2ForCausalLM
        if family == "qwen2"
        else transformers.LlamaForCausalLM
    )
    m = cls(cfg).eval()
    m.config._attn_implementation = "sdpa"
    return m


def test_vendored_kvpress_is_the_wheel_and_loads_only_when_asked():
    for rel, sha in VENDORED.items():
        p = KVDIR / "third_party" / "kvpress" / "kvpress" / rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, rel
    for name in [n for n in sys.modules if n == "kvpress" or n.startswith("kvpress.")]:
        sys.modules.pop(name)
    ea = _ea()
    assert not ea.active() and "kvpress" not in sys.modules
    press, utils = ea.load_kvpress()
    assert (
        press.__name__ == "ExpectedAttentionPress"
        and utils.compute_n_kept(100, 0.75) == 25
    )
    p = press()
    assert (p.n_future_positions, p.n_sink, p.use_covariance, p.use_vnorm) == (
        512,
        4,
        True,
        True,
    )


def test_matched_bytes():
    ea = _ea()
    vbe = ea.value_bits(
        4, 128
    )  # VAL_BITS plus a 16-bit minimum and scale per token, head
    assert vbe == 4 + 32 / 128 and ea.value_bits(4, 128, 32) == 4 + 1
    T, hot, kb = 3000, 128, 3.6
    want = ((T - hot) * (kb + vbe) + hot * 32) / (2 * T)
    assert ea.u3_bits(T, kb, hot, vbe) == pytest.approx(want)
    assert ea.u3_bits(100, None, hot, vbe) == 16.0  # nothing coded: u3 stores fp16
    r = ea.ratio_for(want)
    _, utils = ea.load_kvpress()
    realized = (
        16 * utils.compute_n_kept(T, r) / T
    )  # kvpress keeps int(T (1 - r)) entries
    assert abs(realized / want - 1) < 0.01  # inside the scorer's 1%
    assert 0.7 < r < 0.8  # the prereg's "about 80% evicted" at 3 bits, 4-bit values


def test_eviction_keeps_what_kvpress_keeps():
    ea = _ea(EA="ea", EA_REF="/nonexistent")
    model = _tiny()
    ea.install(model)
    ea.set_prompt(200, 3.5, 128, ea.value_bits(4, 64))
    x = torch.randint(0, 300, (1, 200), generator=torch.Generator().manual_seed(1))
    seen = {}

    def grab(
        mod, args, kwargs, out
    ):  # the cache before the eviction hook (registered last)
        lay = kwargs["past_key_values"].layers[mod.layer_idx]
        if lay.keys.shape[2] == kwargs["hidden_states"].shape[1]:
            seen[mod.layer_idx] = (
                lay.keys.clone(),
                lay.values.clone(),
                kwargs["hidden_states"].clone(),
                dict(kwargs),
            )
        return out

    hooks = [
        lay.self_attn.register_forward_hook(grab, with_kwargs=True, prepend=True)
        for lay in model.model.layers
    ]
    with torch.no_grad():
        out = model.generate(
            x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=6,
            do_sample=False,
            return_dict_in_generate=True,
        )
    for h in hooks:
        h.remove()
    n = ea._S.utils.compute_n_kept(200, ea._S.ratio)
    for li, lay in enumerate(out.past_key_values.layers):
        assert (
            lay.keys.shape[2] == n + 5
        )  # prefill evicted once, then 5 decode steps fp16
        k, v, h, kw = seen[li]
        ref = ea._S.press.__class__(compression_ratio=ea._S.ratio)
        rk, rv = ref.compress(model.model.layers[li].self_attn, h, k, v, None, kw)
        torch.testing.assert_close(lay.keys[:, :, :n], rk, rtol=0, atol=0)
        torch.testing.assert_close(lay.values[:, :, :n], rv, rtol=0, atol=0)
    bits = ea.kv_bits()
    assert bits["arm"] == pytest.approx(16 * n / 200) and bits["tokens"] == 200


@pytest.mark.parametrize("family", ["qwen2", "llama"])
def test_perplexity_mask(family):
    ea = _ea(EA="ea", EA_REF="/nonexistent")
    model = _tiny(family=family)
    ea.install(model)
    x = torch.randint(0, 300, (1, 160), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        plain = model(x, use_cache=True).logits
    T = x.shape[1]
    ea._S.mode, ea._S.ratio = "mask", 0.5
    ea._S.keep = {li: torch.ones(1, 2, T, dtype=torch.bool) for li in range(3)}
    with torch.no_grad():
        full = model(x, use_cache=True).logits
    torch.testing.assert_close(full, plain, rtol=1e-4, atol=1e-4)
    j = 70
    for keep in ea._S.keep.values():
        keep[..., j] = False
    with torch.no_grad():
        cut = model(x, use_cache=True).logits
    torch.testing.assert_close(cut[:, :j], plain[:, :j], rtol=1e-4, atol=1e-4)
    assert (cut[:, j:] - plain[:, j:]).abs().amax(-1).min() > 0
    ea.done()
    ea.set_prompt(T, 3.5, 128, ea.value_bits(4, 64))
    ea.select(model, x)  # pass 1 records kvpress's choice in every layer
    n = ea._S.utils.compute_n_kept(T, ea._S.ratio)
    assert ea._S.mode == "mask" and len(ea._S.keep) == 3
    assert all(
        int(k.sum(-1).min()) == n == int(k.sum(-1).max()) for k in ea._S.keep.values()
    )
    assert all(
        k[..., :4].all() for k in ea._S.keep.values()
    )  # kvpress keeps the 4 sinks


def _pristine_update():
    """transformers' own DynamicCache.update: each harness import in a session wraps the
    previous one's patch, so a second import's coding would run inside the first's."""
    from transformers import DynamicCache

    f = DynamicCache.update
    while f.__name__ == "_patched_update" and "_orig_update" in f.__globals__:
        f = f.__globals__["_orig_update"]
    return f


@pytest.mark.parametrize("mode", ["ea", "ea_u3"])
def test_harness_codes_nothing_under_ea_and_u3_under_ea_u3(mode, tmp_path, monkeypatch):
    from test_key_coding import _harness
    from transformers import DynamicCache

    monkeypatch.setattr(DynamicCache, "update", _pristine_update())
    model = _tiny()
    x = torch.randint(0, 300, (1, 200), generator=torch.Generator().manual_seed(3))
    _ea(EA=mode, EA_REF=str(tmp_path))
    h = _harness(
        CODEBOOK="uniform",
        KEY_BITS=3,
        VAL_BITS=4,
        GROUP=32,
        HOT=128,
        SINK=4,
        OUTLIER_FRAC=0.02,
        PREROPE=0,
        EA=mode,
        EA_REF=str(tmp_path),
    )
    assert h.EA.active() and h.EA.MODE == mode
    h.ensure_basis_calibration(model, None)  # installs the eviction hooks
    ratio = h.EA.set_prompt(200, 3.5, h.HOT, h.EA.value_bits(h.VB, 64, h.VAL_GROUP))
    h._ACCT.clear()
    with torch.no_grad():
        out = model(x, use_cache=True)
    coded = h._bits_summary()
    assert (coded is None) == (mode == "ea")  # ea: u3's coding never ran
    n = h.EA._S.utils.compute_n_kept(200, ratio)
    assert all(lay.keys.shape[2] == n for lay in out.past_key_values.layers)
    bits = h.EA.kv_bits((coded or {}).get("total"))
    if mode == "ea":
        assert bits["arm"] == pytest.approx(16 * n / 200)
    else:  # u3-coded settled entries and fp16 hot ones: fewer bytes than ea
        assert bits["arm"] < 16 * n / 200
    h.EA.remove()


def test_runner_record_and_scorer_check(tmp_path):
    import part5_grid as KG
    import score_part5 as S

    assert KG.MATCHED_BASELINES == ("aatc", "aatc_native", "ea")
    assert set(KG.BASELINES) == {"aatc", "aatc_native", "ea", "ea_u3"}
    assert {"ea", "ea_u3"} <= set(KG.WAVES["baselines"])
    u3 = KG.all_lines()["u3"]
    assert (
        KG.all_lines()["ea"] == u3 + " EA=ea"
        and KG.all_lines()["ea_u3"] == u3 + " EA=ea_u3"
    )
    sys.argv = ["x"]
    import keys_run as R

    for arm, scale in (("ea", 1.0), ("ea_u3", 0.6)):
        cell = tmp_path / arm
        cell.mkdir()
        rows = [
            {
                "idx": i,
                "pred": "",
                "answers": [],
                "all_classes": None,
                "kv_bits": {
                    "arm": 4.0 * scale,
                    "u3": 4.0,
                    "tokens": 1000 + i,
                    "ratio": 0.75,
                },
            }
            for i in range(3)
        ]
        (cell / "qasper.0.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        R.write_ea_baseline(str(cell), arm)
        rec = json.load(open(cell / "baseline.json"))
        key = "matched" if arm == "ea" else "reported"
        assert rec[key]["arm_bits"] == pytest.approx(4.0 * scale)
        why = S.verify_baseline(str(cell), arm)
        assert why is None or why.startswith("incomplete"), why  # the byte check passed


def test_existing_arms_unchanged_since_the_registered_harness():
    """The registered harness is 2e8956a plus Expected Attention (owner, 2026-10-14): in
    every file that produces an existing arm's cells the change adds lines and removes
    none, and in the harness and perplexity script every added hunk is gated on EA."""
    try:
        subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", REGISTERED],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("no git history here")
    files = [
        "tq_paper_lb_shard.py",
        "wikitext_ppl.py",
        "keys_run.py",
        "keys_cell.py",
        "key_coding.py",
        "aatc.py",
        "keys_grid.py",
        "pod_guard.py",
        "part5_cache_fp16.py",
    ]
    for f in files:
        d = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "-U0",
                REGISTERED,
                "--",
                f"benchmarks/kvquant_matrix/{f}",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        removed = [
            ln
            for ln in d.splitlines()
            if ln.startswith("-") and not ln.startswith("---")
        ]
        assert not removed, (f, removed[:3])
        if f in ("tq_paper_lb_shard.py", "wikitext_ppl.py"):
            hunks = d.split("\n@@")[1:]
            for hk in hunks:
                body = "\n".join(ln for ln in hk.splitlines()[1:] if ln.startswith("+"))
                assert "EA" in body or "ea_head_dim" in body, (f, body[:200])
    old = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{REGISTERED}:benchmarks/kvquant_matrix/part5_grid.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ns: dict = {}
    sys.path.insert(0, str(KVDIR))
    exec(compile(old, "part5_grid_2e8956a", "exec"), ns)
    import part5_grid as KG

    now = KG.all_lines()
    for arm, line in ns["all_lines"]().items():
        assert now[arm] == line, arm


def test_v5_judges_ea_on_qasper_alone():
    """Owner, 2026-10-14: Expected Attention's stored-cache perplexity would decide V5's
    perplexity endpoint by construction, so V5 reads qasper alone against ea and both
    endpoints against AATC."""
    import part5_grid as KG
    import score_part5 as S

    assert KG.V5_ENDPOINTS == {"aatc": ("qasper", "ppl"), "ea": ("qasper",)}
    tie_q = [{"qasper": "same", "ppl": "better", "model": m} for m in KG.TIER_A]
    good = [{"qasper": "better", "ppl": "same", "model": m} for m in KG.TIER_A]
    # better on ea's perplexity only: not better against ea
    assert S.v5_rule({"aatc": good, "ea": tie_q}) == "INCONCLUSIVE"
    worse_ppl = [{"qasper": "better", "ppl": "worse", "model": m} for m in KG.TIER_A]
    # worse on ea's perplexity: ignored against ea, counted against AATC
    assert S.v5_rule({"aatc": good, "ea": worse_ppl}) == "HOLDS"
    assert S.v5_rule({"aatc": worse_ppl, "ea": good}) == "FAILS"
