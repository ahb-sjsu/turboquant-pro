"""Part III harness (benchmarks/weight_observer), CPU, tiny models.

Pins what the registration relies on: the codec; fixed-rate, predictor-blind
variants; predictor identities (the competitors are reductions of the observer
form); the statistics against brute-force autograd; the observer form against the
true KL on a toy where its factorization is exact to second order; and the
resumable end-to-end run.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from weight_observer import quant, score, tables, variants  # noqa: E402


def test_rtn_error_falls_with_bits_and_8_bits_is_near_exact():
    w = torch.randn(64, 256, generator=torch.Generator().manual_seed(0))
    errs = [float((quant.rtn(w, b) - w).pow(2).mean()) for b in quant.LEVELS]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-4 * float(w.pow(2).mean())
    assert quant.stored_bits(256, 4) == 256 * 4 + 2 * 32


def test_variants_are_at_their_rate_deterministic_and_blind():
    names = [f"m{i}" for i in range(40)]
    sizes = list(np.random.default_rng(1).integers(1_000, 50_000, 40))
    v1 = variants.generate(names, sizes)
    v2 = variants.generate(names, sizes)
    n_strata = len(variants.RATES) * variants.PER_RATE
    assert v1 == v2 and len(v1) == n_strata + len(variants.CONTROLS)
    s = np.asarray(sizes, float)
    for vid, bits in v1.items():
        if vid.startswith("u"):
            assert set(bits.values()) == {int(vid[1:])}
            continue
        rate = float(vid.split("-")[0][1:])
        mb = variants.mean_bits(np.asarray([bits[n] for n in names]), s)
        assert abs(mb - rate) <= variants.TOL
        assert set(bits.values()) <= set(variants.GEN_LEVELS)
    # variants at one rate differ in where the bits went
    same_rate = [tuple(b.values()) for k, b in v1.items() if k.startswith("r4.0")]
    assert len(set(same_rate)) == len(same_rate)


def test_predictor_identities():
    g = torch.Generator().manual_seed(2)
    w = torch.randn(16, 128, generator=g)
    a = torch.randn(128, 128, generator=g)
    S = a @ a.T / 128
    b = torch.randn(16, 16, generator=g)
    P = b @ b.T / 16
    F = torch.rand(16, 128, generator=g)
    eye = {"S": torch.eye(128), "P": torch.eye(16), "F": F}
    r = tables.predictor_row(w, eye)
    for bits in quant.LEVELS:
        assert r[bits]["act"] == pytest.approx(r[bits]["raw"], rel=1e-5)
        assert r[bits]["observer"] == pytest.approx(r[bits]["act"], rel=1e-5)
    r = tables.predictor_row(w, {"S": S, "P": torch.eye(16), "F": F})
    assert r[3]["observer"] == pytest.approx(r[3]["act"], rel=1e-5)
    Pd = torch.diag(torch.diagonal(P))
    r = tables.predictor_row(w, {"S": S, "P": Pd, "F": F})
    assert r[3]["observer"] == pytest.approx(r[3]["outdiag"], rel=1e-5)


class _Toy(torch.nn.Module):
    """logits = W x: one linear layer read directly by a softmax."""

    def __init__(self, d_in=32, vocab=24, seed=0):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, vocab, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(
                0.05
                * torch.randn(
                    vocab, d_in, generator=torch.Generator().manual_seed(seed)
                )
            )

    def forward(self, x):
        return self.lin(x)


def test_statistics_match_brute_force_autograd():
    toy = _Toy()
    toy.requires_grad_(False)
    g = torch.Generator().manual_seed(3)
    xs = [torch.randn(1, 10, 32, generator=g) for _ in range(4)]
    acc = tables.Accumulator({"lin": toy.lin})
    gen = torch.Generator().manual_seed(4)
    labels = []
    for x in xs:
        x = x.clone().requires_grad_(True)
        logp = torch.log_softmax(toy(x), -1)
        y = torch.multinomial(logp.exp().reshape(-1, 24).detach(), 1, generator=gen)
        labels.append(y)
        (-logp.reshape(-1, 24).gather(1, y).sum()).backward()
    acc.close()
    st = acc.finalized()["lin"]
    X = torch.cat([x.reshape(-1, 32) for x in xs])
    assert torch.allclose(st["S"], (X.T @ X / X.shape[0]).float(), atol=1e-5)
    assert torch.allclose(st["A"], X.abs().mean(0).float(), atol=1e-6)
    # brute force: weight gradient per sequence, squared, averaged
    Fb = torch.zeros(24, 32)
    Pb = torch.zeros(24, 24)
    for x, y in zip(xs, labels):
        w = toy.lin.weight.detach().clone().requires_grad_(True)
        z = x.reshape(-1, 32) @ w.T
        z.retain_grad()
        (-torch.log_softmax(z, -1).gather(1, y).sum()).backward()
        Fb += w.grad**2
        Pb += z.grad.T @ z.grad
    assert torch.allclose(st["F"], Fb / 4, rtol=1e-4, atol=1e-7)
    assert torch.allclose(st["P"], Pb / 40, rtol=1e-4, atol=1e-7)


def test_observer_tracks_the_true_kl_where_its_factorization_holds():
    """Near-uniform softmax: P_y is the same at every input, so the K-FAC form is exact
    to second order and KL = (1/2) tr(P D Sigma D^T); raw distortion ignores the reader.
    """
    toy = _Toy(seed=5)
    toy.requires_grad_(False)
    g = torch.Generator().manual_seed(6)
    X = torch.randn(1, 4000, 32, generator=g)
    acc = tables.Accumulator({"lin": toy.lin})
    gen = torch.Generator().manual_seed(7)
    x = X.clone().requires_grad_(True)
    logp = torch.log_softmax(toy(x), -1)
    y = torch.multinomial(logp.exp().reshape(-1, 24).detach(), 1, generator=gen)
    (-logp.reshape(-1, 24).gather(1, y).sum()).backward()
    acc.close()
    st = acc.finalized()["lin"]
    w0 = toy.lin.weight.detach().clone()
    for s in range(5):
        d = 0.02 * torch.randn(24, 32, generator=torch.Generator().manual_seed(100 + s))
        with torch.no_grad():
            lp0 = torch.log_softmax(X.reshape(-1, 32) @ w0.T, -1)
            lp1 = torch.log_softmax(X.reshape(-1, 32) @ (w0 + d).T, -1)
            kl = float((lp0.exp() * (lp0 - lp1)).sum(-1).mean())
        pred = 0.5 * float(((st["P"] @ d) * (d @ st["S"])).sum())
        assert pred == pytest.approx(kl, rel=0.15)


def test_scorer_spearman_and_verdict_rules():
    assert score.spearman(
        np.array([1, 2, 3, 4.0]), np.array([10, 20, 30, 40.0])
    ) == pytest.approx(1)
    assert score.spearman(
        np.array([1, 2, 3, 4.0]), np.array([4, 3, 2, 1.0])
    ) == pytest.approx(-1)

    def rep(lo, hi, pt):
        comps = ("fisher", "act", "raw", "outdiag")
        return {"diff": {c: {"lo": lo, "hi": hi, "point": pt} for c in comps}}

    assert (
        score.verdicts({m: rep(0.01, 0.1, 0.05) for m in score.MODELS})["W1"] == "HOLDS"
    )
    assert (
        score.verdicts({m: rep(-0.1, -0.01, -0.05) for m in score.MODELS})["W1"]
        == "FAILS (reversed)"
    )
    assert (
        score.verdicts({m: rep(-0.02, 0.05, 0.01) for m in score.MODELS})["W1"]
        == "INCONCLUSIVE"
    )
    assert score.verdicts({score.MODELS[0]: rep(0.01, 0.1, 0.05)})["W1"] == "INCOMPLETE"


def test_end_to_end_on_a_tiny_llama(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from weight_observer import run as R

    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg)
    mdir = tmp_path / "model"
    model.save_pretrained(mdir)

    class _Tok:
        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([ord(c) % 128 for c in text])
            return type("E", (), {"input_ids": ids[None]})()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: _Tok())
    monkeypatch.setattr(R, "N_CALIB", 3)
    monkeypatch.setattr(R, "N_EVAL", 4)
    monkeypatch.setattr(R, "SEQ", 64)
    monkeypatch.setattr(variants, "PER_RATE", 2)
    text = tmp_path / "text"
    text.mkdir()
    (text / "train.txt").write_text("the quick brown fox jumps over the lazy dog " * 40)
    (text / "test.txt").write_text("pack my box with five dozen liquor jugs " * 40)
    out = tmp_path / "out"
    args = [
        "--model-key",
        "tiny",
        "--model-path",
        str(mdir),
        "--text",
        str(text),
        "--out",
        str(out),
        "--device",
        "cpu",
    ]
    assert R.main(args) == 0
    rows = [json.loads(line) for line in open(out / "results.jsonl")]
    assert len(rows) == len(variants.RATES) * 2 + len(variants.CONTROLS)
    u8 = next(r for r in rows if r["variant"] == "u8")
    u3 = next(r for r in rows if r["variant"] == "u3")
    assert sum(s["kl_sum"] for s in u8["seqs"]) < sum(s["kl_sum"] for s in u3["seqs"])
    for r in rows:
        assert set(r["pred"]) == set(tables.PREDICTORS) and all(
            v >= 0 for v in r["pred"].values()
        )
        assert all(s["kl_sum"] >= -1e-6 for s in r["seqs"])
    atlas = json.load(open(out / "atlas.json"))["atlas"]
    assert len(atlas) == 14 and all("row_hubness" in a for a in atlas.values())
    # resumable: a second call measures nothing new
    assert R.main(args) == 0
    assert len(open(out / "results.jsonl").readlines()) == len(rows)


def test_exploratory_exact_forms_on_a_tiny_llama(tmp_path, monkeypatch):
    """explore.py: the registered fisher table is reproduced (wiring), c has one row per
    calibration sequence, and exact_model differs from exact_block only by cross terms.
    """
    transformers = pytest.importorskip("transformers")
    from weight_observer import explore as X
    from weight_observer import explore_score as XS

    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).eval()
    model.requires_grad_(False)
    g = torch.Generator().manual_seed(1)
    calib = [torch.randint(0, 128, (1, 32), generator=g) for _ in range(3)]
    ex = X.build(model, calib, group_size=1, seed=7, log=lambda s: None)
    tbl, _ = tables.build(model, calib, 1, 7, log=lambda s: None)
    for n in ex["names"]:
        for i, b in enumerate(ex["levels"]):
            assert ex["fisher_check"][n][i] == pytest.approx(
                tbl[n][b]["fisher"], rel=1e-3
            )
    c = ex.pop("c")
    assert c.shape == (3, len(ex["names"]), len(ex["levels"]))
    bits = {n: 4 for n in ex["names"]}
    p = XS.predictors(ex, c, bits)
    li = ex["levels"].index(4)
    cs = c[:, :, li]
    cross = ((cs.sum(1) ** 2) - (cs**2).sum(1)).mean()
    assert p["exact_model"] - p["exact_block"] == pytest.approx(
        cross, rel=1e-9, abs=1e-12
    )


def test_sensitivity_sweep_and_oracle_on_a_tiny_llama(tmp_path, monkeypatch):
    """sensitivity.py measures every (matrix, bits) once, restores the matrix after each
    (a second pass over the finished file adds nothing), and the oracle sums the lines.
    """
    transformers = pytest.importorskip("transformers")
    from weight_observer import explore_score as XS
    from weight_observer import run as R
    from weight_observer import sensitivity as S

    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    mdir = tmp_path / "model"
    transformers.LlamaForCausalLM(cfg).save_pretrained(mdir)

    class _Tok:
        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([ord(c) % 128 for c in text])
            return type("E", (), {"input_ids": ids[None]})()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: _Tok())
    monkeypatch.setattr(R, "N_EVAL", 2)
    monkeypatch.setattr(R, "SEQ", 64)
    text = tmp_path / "text"
    text.mkdir()
    (text / "test.txt").write_text("pack my box with five dozen liquor jugs " * 40)
    out = tmp_path / "out"
    args = ["--model-path", str(mdir), "--text", str(text), "--out", str(out)]
    assert S.main([*args, "--device", "cpu"]) == 0
    lines = (out / "sensitivity.jsonl").read_text().splitlines()
    assert len(lines) == 7 * len(S.LEVELS)
    assert S.main([*args, "--device", "cpu"]) == 0
    assert len((out / "sensitivity.jsonl").read_text().splitlines()) == len(lines)
    sens = XS.single_kl(str(out / "sensitivity.jsonl"))
    names = sorted({m for m, _ in sens})
    kl3 = [sens[(n, 3)] for n in names]
    kl6 = [sens[(n, 6)] for n in names]
    assert all(a > b >= 0 for a, b in zip(kl3, kl6))
    bits = {n: (3 if i % 2 else 8) for i, n in enumerate(names)}
    want = sum(sens[(n, 3)] for i, n in enumerate(names) if i % 2)
    assert XS.oracle_add(sens, bits) == pytest.approx(want)


def test_plans_eval_only_measures_the_named_predictors(tmp_path, monkeypatch):
    """``plans eval --only`` measures the named predictors' plans and nothing else,
    into its own output, and a second pass adds nothing; the NRP oracle check uses it
    that way."""
    transformers = pytest.importorskip("transformers")
    from weight_observer import nrp
    from weight_observer import plans as PL
    from weight_observer import run as R

    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    mdir = tmp_path / "model"
    model = transformers.LlamaForCausalLM(cfg)
    model.save_pretrained(mdir)
    names = sorted(tables.linear_modules(model))

    class _Tok:
        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([ord(c) % 128 for c in text])
            return type("E", (), {"input_ids": ids[None]})()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: _Tok())
    monkeypatch.setattr(R, "N_EVAL", 2)
    monkeypatch.setattr(R, "SEQ", 64)
    text = tmp_path / "text"
    text.mkdir()
    (text / "test.txt").write_text("pack my box with five dozen liquor jugs " * 40)
    plans = {
        f"p4.0-{p}": {n: b for n in names}
        for p, b in (("fisher", 4), ("oracle", 5), ("raw", 3))
    }
    pf = tmp_path / "plans.json"
    pf.write_text(json.dumps({"plans": plans}))
    out = tmp_path / "oracle_check"
    args = ["eval", "--model-path", str(mdir), "--text", str(text), "--plans", str(pf)]
    args += ["--out", str(out), "--device", "cpu", "--only", "oracle,fisher"]
    assert PL.main(args) == 0
    lines = (out / "plans_results.jsonl").read_text().splitlines()
    got = [json.loads(x)["plan"] for x in lines]
    assert sorted(got) == ["p4.0-fisher", "p4.0-oracle"]
    assert PL.main(args) == 0
    assert len((out / "plans_results.jsonl").read_text().splitlines()) == len(got)
    s = nrp.oracle_script("a" * 40, "qwen2.5-1.5b")
    assert "--only oracle,fisher,exact_block,fisher_tok" in s and "oracle_check" in s
    assert " sleep" not in s


def test_flatness_swaps_keep_the_budget_exactly_and_are_seeded():
    """A flatness perturbation swaps widths between same-type matrices only, so stored
    bytes are unchanged exactly; k swaps change 2k matrices; the plans are seeded."""
    from weight_observer import flatness as FL
    from weight_observer import nrp

    rng = np.random.default_rng(0)
    kinds = ("self_attn.q_proj", "mlp.up_proj")
    names = [f"layers.{i}.{p}" for i in range(12) for p in kinds]
    numel = {n: (4096 if n.endswith("q_proj") else 11008) for n in names}
    plan = {n: int(rng.choice([3, 4, 5, 6, 8])) for n in names}
    stored = sum(numel[n] * plan[n] for n in names)
    for k in (1, 4, 8):
        out, moved = FL.perturb(plan, numel, k, np.random.default_rng(k))
        assert sum(numel[n] * out[n] for n in names) == stored
        assert sum(out[n] != plan[n] for n in names) == 2 * k
        assert all(
            sorted(out[n] for n in names if n.endswith(t))
            == sorted(plan[n] for n in names if n.endswith(t))
            for t in ("q_proj", "up_proj")
        )
        assert 0 < moved < 1
        again, _ = FL.perturb(plan, numel, k, np.random.default_rng(k))
        assert again == out
    with pytest.raises(ValueError, match="no swap left"):
        FL.perturb({n: 4 for n in names}, numel, 1, rng)
    s = nrp.flat_script("a" * 40, "qwen2.5-1.5b")
    assert "flatness.json" in s and "/flatness" in s and " sleep" not in s


def _correlated(n=4096, d=256, seed=11):
    """Inputs with correlated channels and a few large ones (what GPTQ and AWQ use)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    x = z @ (torch.eye(d) + 0.3 * torch.randn(d, d, generator=g) / d**0.5)
    x[:, :4] *= 20.0
    return x, (x.T @ x / n), x.abs().mean(0)


@pytest.mark.parametrize("bits", quant.LEVELS)
def test_g0_gptq_with_identity_hessian_is_rtn_bit_for_bit(bits):
    """G0 (Part III-c): with H = I and no damping nothing is pushed forward, so GPTQ is
    RTN exactly: the codecs share one grid and one rounding rule."""
    w = torch.randn(32, 384, generator=torch.Generator().manual_seed(bits))
    got = quant.gptq(w, torch.eye(384), bits, damp=0.0)
    assert torch.equal(got, quant.rtn(w, bits))


@pytest.mark.parametrize("bits", quant.LEVELS)
def test_g0_awq_with_alpha_zero_is_rtn_bit_for_bit(bits):
    w = torch.randn(32, 384, generator=torch.Generator().manual_seed(bits))
    _, S, a = _correlated(d=384)
    got, alpha = quant.awq(w, S, a, bits, alphas=(0.0,))
    assert alpha == 0.0 and torch.equal(got, quant.rtn(w, bits))


def test_gptq_lowers_the_output_error_it_targets():
    w = torch.randn(64, 256, generator=torch.Generator().manual_seed(1))
    _, S, _ = _correlated()
    for bits in (2, 3, 4):
        e_rtn = quant.output_error(quant.rtn(w, bits) - w, S)
        e_gptq = quant.output_error(quant.gptq(w, S, bits) - w, S)
        assert e_gptq < 0.9 * e_rtn, (bits, e_gptq, e_rtn)


def test_gptq_lazy_block_updates_match_a_single_block():
    """Block-wise (lazy) error propagation equals doing every column in one block."""
    w = torch.randn(16, 512, generator=torch.Generator().manual_seed(2))
    _, S, _ = _correlated(d=512)
    a = quant.gptq(w, S, 3, block=128)
    b = quant.gptq(w, S, 3, block=512)
    assert torch.allclose(a, b, atol=1e-5)


def test_awq_never_does_worse_than_rtn_and_helps_with_large_channels():
    """alpha = 0 is in the grid and the choice minimizes the calibration output error,
    so AWQ is never worse than RTN on it; with a few large input channels it is
    better."""
    w = torch.randn(64, 256, generator=torch.Generator().manual_seed(3))
    _, S, a = _correlated()
    wq, alpha = quant.awq(w, S, a, 3)
    e_awq = quant.output_error(wq - w, S)
    e_rtn = quant.output_error(quant.rtn(w, 3) - w, S)
    assert e_awq <= e_rtn and alpha > 0 and e_awq < 0.9 * e_rtn


def test_codec_run_end_to_end_on_a_tiny_llama(tmp_path, monkeypatch):
    """Part III-c harness: tables -> plans -> arms. Every arm within its budget (G1),
    uniform arms exactly at it; the statistics are deterministic, so a cost and the arm
    measured later see the same codec output; the uniform RTN arm measures exactly what
    Part III's own path does; every phase resumes without repeating work."""
    transformers = pytest.importorskip("transformers")
    from weight_observer import codec_run as CR
    from weight_observer import run as R
    from weight_observer.measure import apply_variant, kl_per_sequence

    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    mdir = tmp_path / "model"
    transformers.LlamaForCausalLM(cfg).save_pretrained(mdir)

    class _Tok:
        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([ord(c) % 128 for c in text])
            return type("E", (), {"input_ids": ids[None]})()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: _Tok())
    monkeypatch.setattr(R, "SEQ", 64)
    monkeypatch.setattr(CR, "N_CALIB", 3)
    monkeypatch.setattr(CR, "N_EVAL", 2)
    text = tmp_path / "text"
    text.mkdir()
    (text / "train.txt").write_text("the quick brown fox jumps over the lazy dog " * 40)
    (text / "test.txt").write_text("pack my box with five dozen liquor jugs " * 40)
    out = tmp_path / "out"
    base = ["--model-path", str(mdir), "--text", str(text), "--out", str(out)]
    dev = ["--device", "cpu"]

    assert CR.main(["tables", *base, *dev]) == 0
    rows = (out / "codec_costs.jsonl").read_text().splitlines()
    assert len(rows) == 14  # 2 layers x 7 matrices
    assert CR.main(["tables", *base, *dev]) == 0  # resumes: nothing repeated
    assert len((out / "codec_costs.jsonl").read_text().splitlines()) == 14
    h = json.loads((out / "hashes.json").read_text())
    assert h["evaluation_windows"]["n"] == 2 and h["calibration_windows"]["n"] == 3

    assert CR.main(["plans", "--out", str(out)]) == 0
    spec = json.loads((out / "arms.json").read_text())
    assert len(spec) == 2 * 8
    for arm, v in spec.items():
        assert v["stored_bits"] <= v["budget_bits"], arm
        if "_u" in arm:
            assert v["stored_bits"] == v["budget_bits"], arm

    ref = R.load(str(mdir), "cpu")
    mods = tables.linear_modules(ref)
    calib = [c[None] for c in R.chunks(_Tok(), (text / "train.txt").read_text(), 3)]
    first_two = dict(list(mods.items())[:2])
    one = CR.input_stats(ref, first_two, calib)
    two = CR.input_stats(ref, first_two, calib)
    assert all(torch.equal(one[n]["S"], two[n]["S"]) for n in one)

    only = ["--only", "rtn_u4,gptq_u4,awq_f3,gptq_seq_u3"]
    assert CR.main(["arms", *base, *dev, *only]) == 0
    res = {
        json.loads(x)["arm"]: json.loads(x)["seqs"]
        for x in (out / "arms_results.jsonl").read_text().splitlines()
    }
    assert set(res) == {"rtn_u4", "gptq_u4", "awq_f3", "gptq_seq_u3"}
    assert CR.main(["arms", *base, *dev, *only]) == 0  # resumes
    assert len((out / "arms_results.jsonl").read_text().splitlines()) == 4

    var = R.load(str(mdir), "cpu")
    evalq = [c[None] for c in R.chunks(_Tok(), (text / "test.txt").read_text(), 2)]
    apply_variant(ref, var, {n: 4 for n in mods})
    part3 = kl_per_sequence(ref, var, evalq)
    assert [s["kl_sum"] for s in part3] == [s["kl_sum"] for s in res["rtn_u4"]]


def _fake_results(root, models, kl_of, repeat=None):
    """A results tree: per model, two matrices, the eight arm families at both budgets,
    per-sequence KL from kl_of(arm) (48 sequences)."""
    from weight_observer import quant as Qm

    numel = {"a": 1024, "b": 2048}
    for m in models:
        d = root / m
        d.mkdir(parents=True)
        (d / "codec_costs.jsonl").write_text(
            chr(10).join(
                json.dumps({"matrix": n, "numel": k}) for n, k in numel.items()
            )
        )
        arms, lines = {}, []
        for b in (3, 4):
            for fam in (
                "rtn_u",
                "rtn_f",
                "gptq_u",
                "gptq_f",
                "awq_u",
                "awq_f",
                "gptq_frtn",
                "gptq_seq_u",
            ):
                arm = f"{fam}{b}"
                mb = 8 if "_f" in fam else 0
                bits = {n: b for n in numel}
                st = sum(Qm.stored_bits(k, b) for k in numel.values()) + mb * 2
                arms[arm] = {
                    "codec": fam.split("_")[0],
                    "budget": b,
                    "bits": bits,
                    "stored_bits": st,
                    "budget_bits": st,
                    "map_bits": mb,
                }
                kl = kl_of(m, arm)
                lines.append(
                    json.dumps(
                        {
                            "arm": arm,
                            "seqs": [
                                {"kl_sum": float(v) * 1024, "tokens": 1024} for v in kl
                            ],
                        }
                    )
                )
        (d / "arms.json").write_text(json.dumps(arms))
        (d / "arms_results.jsonl").write_text(chr(10).join(lines))
        if repeat:
            (d / "arms_repeat.jsonl").write_text(repeat(m, lines))


def test_score_codec_judges_in_noise_and_size_and_applies_the_rules(tmp_path):
    from weight_observer import score_codec as SC

    rng = np.random.default_rng(0)
    base = 0.1 + 0.02 * rng.random(48)

    def kl_of(m, arm):  # gptq_f 20% below everything on two models, equal on the third
        f = 0.8 if arm.startswith("gptq_f") and not arm.startswith("gptq_frtn") else 1.0
        return base * (1.0 if m == SC.MODELS[2] else f)

    _fake_results(tmp_path / "r", SC.MODELS, kl_of)
    r = SC.score(str(tmp_path / "r"))
    assert {m: g["status"] for m, g in r["gates"]["G1"].items()} == dict.fromkeys(
        SC.MODELS, "PASS"
    )
    assert r["verdict_status"] == "PROVISIONAL"  # samples and G2 not yet checkable
    assert r["verdicts"] == dict.fromkeys(("C1a", "C1b", "C2", "C3"), "HOLDS")
    c = r["comparisons"][f"{SC.MODELS[0]}|gptq_f3|gptq_u3"]
    assert c["rel"] == pytest.approx(-0.2) and c["judgement"] == "better"
    same = r["comparisons"][f"{SC.MODELS[2]}|gptq_f3|gptq_u3"]
    assert same["rel"] == 0 and same["judgement"] == "neither"  # paired: identical arms
    assert SC.verdict({SC.MODELS[0]: {3: "better", 4: "better"}}) == "INCOMPLETE"
    worse = {m: {3: "worse", 4: "neither"} for m in SC.MODELS}
    assert SC.verdict(worse) == "FAILS (reversed)"
    small = {m: {3: "neither", 4: "neither"} for m in SC.MODELS}
    assert SC.verdict(small) == "FAILS"
    # a 3% effect with a tight interval is still not "better": the 5% bar
    j = SC.compare(base * 0.97, base, np.random.default_rng(1))
    assert j["hi"] < 0 and j["judgement"] == "neither"


def test_score_codec_gates_g1_recomputes_and_g2_withholds(tmp_path):
    from weight_observer import score_codec as SC

    base = 0.1 + 0.01 * np.arange(48) / 48

    def rep_bad(m, lines):
        r = json.loads(lines[0])
        r["seqs"][0]["kl_sum"] += 1.0  # another pod measured something else
        return json.dumps(r)

    _fake_results(tmp_path / "r", SC.MODELS, lambda m, a: base, repeat=rep_bad)
    arms_p = tmp_path / "r" / SC.MODELS[0] / "arms.json"
    arms = json.loads(arms_p.read_text())
    arms["gptq_f3"]["stored_bits"] -= 8  # recorded size no longer matches the widths
    arms_p.write_text(json.dumps(arms))
    r = SC.score(str(tmp_path / "r"))
    assert r["gates"]["G1"][SC.MODELS[0]]["status"] == "FAIL_UNEXPLAINED"
    assert r["gates"]["G2"][SC.MODELS[1]]["status"] == "FAIL_UNEXPLAINED"
    assert r["verdict_status"] == "WITHHELD"
    assert set(r["verdicts"].values()) == {"WITHHELD"}


def test_g0_holds_for_a_stack_of_matrices_at_different_widths():
    """Stacked GPTQ with H = I and no damping: each matrix of the stack, at its own
    width, is RTN bit for bit."""
    g = torch.Generator().manual_seed(21)
    ws = [torch.randn(r, 256, generator=g) for r in (8, 16, 24)]
    got = quant.gptq_stack(ws, torch.eye(256), [2, 4, 8], damp=0.0)
    for w, b, q in zip(ws, (2, 4, 8), got):
        assert torch.equal(q, quant.rtn(w, b))


def test_a_stack_is_the_matrices_encoded_one_by_one():
    """Rows are independent given H: stacking changes only the arithmetic's grouping."""
    g = torch.Generator().manual_seed(22)
    ws = [torch.randn(r, 256, generator=g) for r in (8, 16)]
    _, S, _ = _correlated()
    got = quant.gptq_stack(ws, S, [3, 4])
    for w, b, q in zip(ws, (3, 4), got):
        assert torch.allclose(q, quant.gptq(w, S, b), atol=1e-5)


def test_the_harness_encodes_one_unit_the_same_way_every_time():
    """tables costs every width of a unit and arms picks one; both call encode_unit,
    which is deterministic, and ``want`` changes no RTN or AWQ output."""
    from weight_observer import codec_run as CR

    g = torch.Generator().manual_seed(23)
    _, S, a = _correlated()
    ws = {
        f"layers.0.self_attn.{k}_proj": torch.randn(16, 256, generator=g) for k in "qkv"
    }
    st = {n: {"S": S, "A": a} for n in ws}
    assert CR.units(list(ws)) == [list(ws)]
    for codec in CR.CODECS:
        one, two = CR.encode_unit(codec, ws, st), CR.encode_unit(codec, ws, st)
        want = {n: 3 for n in ws}
        three = CR.encode_unit(codec, ws, st, want)
        for n in ws:
            for b in CR.LEVELS:
                assert torch.equal(one[n][b][0], two[n][b][0])
            assert torch.equal(three[n][3][0], one[n][3][0])


def test_nrp_codec_scripts_run_the_pinned_harness_and_never_sleep():
    from weight_observer import nrp

    t = nrp.ctables_script("a" * 40, "qwen2.5-0.5b")
    r = nrp.carms_script("a" * 40, "qwen2.5-0.5b")
    assert "codec_run tables" in t and "/codec/qwen2.5-0.5b" in t
    assert "codec_run arms" in r and "planned/qwen2.5-0.5b.codec_arms.json" in r
    assert all(" sleep" not in x and "pip install" not in x for x in (t, r))
    assert "cd /data/wo/codec/qwen2.5-0.5b" in nrp.fetch_script("qwen2.5-0.5b", "codec")
    with pytest.raises(ValueError):
        nrp.fetch_script("qwen2.5-0.5b", "elsewhere")
