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
