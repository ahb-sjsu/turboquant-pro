"""The product weight encoder (turboquant_pro.weight_codec, tqp plan encode-weights).

Pins the claim the module makes: it writes the weights Part III-c measured. The codec is
the harness codec bit for bit, G0 holds, and on a tiny Llama the product's in-place,
last-to-first encoding equals the harness's two-model encoding of the same plan exactly.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")

from turboquant_pro import weight_codec as C  # noqa: E402
from turboquant_pro.cli import main as cli_main  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))


def _correlated(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4 * n, n, generator=g) @ torch.randn(n, n, generator=g)
    return (x.T @ x) / x.shape[0]


def test_the_port_is_the_harness_codec_bit_for_bit():
    from weight_observer import quant as Q

    assert (C.GROUP, C.LEVELS) == (Q.GROUP, Q.LEVELS)
    g = torch.Generator().manual_seed(1)
    ws = [torch.randn(r, 256, generator=g) for r in (8, 16, 24)]
    S = _correlated()
    for b in C.LEVELS:
        assert torch.equal(C.rtn(ws[0], b), Q.rtn(ws[0], b))
    got = C.gptq_stack(ws, S, [2, 4, 8])
    want = Q.gptq_stack(ws, S, [2, 4, 8], damp=0.01)
    assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_g0_gptq_with_identity_hessian_is_rtn(bits):
    w = torch.randn(32, 384, generator=torch.Generator().manual_seed(bits))
    assert torch.equal(C.gptq(w, torch.eye(384), bits, damp=0.0), C.rtn(w, bits))


def test_gptq_beats_rtn_on_the_error_it_targets():
    w = torch.randn(64, 256, generator=torch.Generator().manual_seed(2))
    S = _correlated(seed=3)

    def err(q):
        d = (q - w).double()
        return float(((d @ S.double()) * d).sum())

    for b in (2, 3, 4):
        assert err(C.gptq(w, S, b)) < 0.9 * err(C.rtn(w, b))


# ----------------------------------------------------------------------------- model


class _Tok:
    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([ord(c) % 128 for c in text])
        return type("E", (), {"input_ids": ids[None]})()

    def save_pretrained(self, path):
        pass


TEXT = "the quick brown fox jumps over the lazy dog " * 40


@pytest.fixture
def tiny(tmp_path):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,  # two layer groups, so last-to-first order is exercised
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    mdir = tmp_path / "model"
    transformers.LlamaForCausalLM(cfg).save_pretrained(mdir)
    return mdir


def _plan(names, seed=5):
    g = torch.Generator().manual_seed(seed)
    pick = torch.randint(len(C.LEVELS), (len(names),), generator=g)
    return {
        "schema": "tqp.weight_plan/1",
        "bits": {n: C.LEVELS[i] for n, i in zip(names, pick)},
    }


@pytest.mark.parametrize("codec", ["gptq", "rtn"])
def test_encode_model_writes_exactly_what_the_harness_measured(tiny, codec):
    from weight_observer import codec_run as CR
    from weight_observer import run as R
    from weight_observer import tables as T

    calib = C.calibration_windows(_Tok(), TEXT, 3, 64)
    ref = R.load(str(tiny), "cpu")
    var = R.load(str(tiny), "cpu")
    rm, vm = T.linear_modules(ref), T.linear_modules(var)
    plan = _plan(list(rm))
    spec = {"codec": codec, "bits": plan["bits"]}
    for grp in T.layer_groups(len(ref.model.layers), R.GROUP_LAYERS):
        names = [n for n in rm if int(n.split(".")[1]) in grp]
        CR.encode_group(ref, var, rm, vm, names, spec, calib)

    model = R.load(str(tiny), "cpu")
    C.encode_model(model, plan, calib, codec)
    got = C.linear_modules(model)
    assert list(got) == list(vm)
    for n in vm:
        assert torch.equal(got[n].weight, vm[n].weight), n
    assert any(not torch.equal(got[n].weight, rm[n].weight) for n in rm)


def test_encode_model_refuses_a_plan_for_another_model(tiny):
    from weight_observer import run as R

    model = R.load(str(tiny), "cpu")
    names = list(C.linear_modules(model))
    with pytest.raises(ValueError, match="does not match"):
        C.encode_model(model, _plan(names[:-1]), None, "rtn")
    bad = _plan(names)
    bad["bits"][names[0]] = 7
    with pytest.raises(ValueError, match="widths outside"):
        C.encode_model(model, bad, None, "rtn")
    with pytest.raises(ValueError, match="calibration"):
        C.encode_model(model, _plan(names), None, "gptq")


def test_cli_encode_weights_end_to_end(tiny, tmp_path, monkeypatch):
    from weight_observer import run as R

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: _Tok())
    names = list(C.linear_modules(R.load(str(tiny), "cpu")))
    plan = _plan(names)
    plan["codec"] = "gptq"
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(plan))
    txt = tmp_path / "train.txt"
    txt.write_text(TEXT)
    out = tmp_path / "enc"
    base = ["plan", "encode-weights", "--plan", str(pp), "--model-path", str(tiny)]
    base += ["--device", "cpu", "--out", str(out)]

    assert cli_main(base) == 2  # gptq without calibration text
    argv = base + ["--calib-text", str(txt), "--n-calib", "3", "--seq", "64"]
    assert cli_main(argv) == 0
    man = json.loads((out / "weight_encoding.json").read_text())
    assert man["codec"] == "gptq" and man["matrices"] == len(names)
    assert man["calibration"]["windows"] == 3
    assert man["calibration"]["windows_sha256"] == C.windows_sha(
        C.calibration_windows(_Tok(), TEXT, 3, 64)
    )

    want = R.load(str(tiny), "cpu")
    C.encode_model(want, plan, C.calibration_windows(_Tok(), TEXT, 3, 64), "gptq")
    got = R.load(str(out), "cpu")
    wm, gm = C.linear_modules(want), C.linear_modules(got)
    assert all(torch.equal(wm[n].weight, gm[n].weight) for n in wm)

    assert cli_main(argv + ["--n-calib", "999"]) == 2  # too little text, refused


def test_cli_plan_weights_records_the_codec_and_the_schema_accepts_it(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    from turboquant_pro import weight_plan as W
    from turboquant_pro.schemas import load_schema

    v = jsonschema.Draft202012Validator(load_schema("weight_plan.schema.json"))
    mats = {f"m{i}": {"numel": 128 * (i + 1), "group": 128} for i in range(4)}
    costs = {
        n: {b: (i + 1) * 4.0 ** (-b) for b in (3, 4, 8)} for i, n in enumerate(mats)
    }
    cp = tmp_path / "costs.json"
    cp.write_text(json.dumps(W.CostTable("toy", "fisher", mats, costs, {}).as_dict()))
    for extra, codec in (([], "gptq"), (["--codec", "rtn"], "rtn")):
        out = tmp_path / "plan.json"
        argv = ["plan", "weights", "--costs", str(cp), "--bits-per-weight", "4"]
        assert cli_main(argv + extra + ["--out", str(out)]) == 0
        doc = json.loads(out.read_text())
        assert doc["codec"] == codec
        v.validate(doc)
    doc["codec"] = "awq"
    assert not v.is_valid(doc)
    del doc["codec"]
    v.validate(doc)  # plans written before the field still validate


def test_the_product_path_from_harness_costs_to_encoded_weights(
    tiny, tmp_path, monkeypatch
):
    """The registered consequence end to end: the harness's Fisher costs, the one RTN
    table written by ``cost-table``, ``tqp plan weights`` (codec gptq by default), and
    ``tqp plan encode-weights`` writing that plan into the model, every matrix touched.
    """
    from weight_observer import codec_run as CR
    from weight_observer import run as R

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: _Tok())
    monkeypatch.setattr(R, "SEQ", 64)
    monkeypatch.setattr(CR, "N_CALIB", 3)
    monkeypatch.setattr(CR, "N_EVAL", 2)
    text = tmp_path / "text"
    text.mkdir()
    (text / "train.txt").write_text(TEXT)
    (text / "test.txt").write_text("pack my box with five dozen liquor jugs " * 40)
    out = tmp_path / "h"
    base = ["--model-path", str(tiny), "--text", str(text), "--out", str(out)]
    assert CR.main(["tables", *base, "--device", "cpu"]) == 0
    assert CR.main(["cost-table", "--out", str(out), "--model-key", "tiny"]) == 0
    table = out / "cost_table_rtn.json"
    assert json.loads(table.read_text())["predictor"] == "fisher/rtn"

    plan = tmp_path / "plan.json"
    argv = ["plan", "weights", "--costs", str(table), "--bits-per-weight", "3.5"]
    assert cli_main(argv + ["--out", str(plan)]) == 0
    doc = json.loads(plan.read_text())
    assert doc["codec"] == "gptq" and len(set(doc["bits"].values())) > 1

    enc = tmp_path / "enc"
    argv = ["plan", "encode-weights", "--plan", str(plan), "--model-path", str(tiny)]
    argv += ["--calib-text", str(text / "train.txt"), "--n-calib", "3", "--seq", "64"]
    assert cli_main(argv + ["--device", "cpu", "--out", str(enc)]) == 0
    before = C.linear_modules(R.load(str(tiny), "cpu"))
    after = C.linear_modules(R.load(str(enc), "cpu"))
    assert all(not torch.equal(before[n].weight, after[n].weight) for n in before)
    man = json.loads((enc / "weight_encoding.json").read_text())
    assert man["cost_table_hash"] == doc["cost_table_hash"]
