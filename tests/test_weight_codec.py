"""The product weight encoder (turboquant_pro.weight_codec, tqp plan encode-weights).

Pins the claim the module makes: it writes the weights Part III-c measured. The codec is
the harness codec bit for bit, G0 holds, and on a tiny Llama the product's in-place,
last-to-first encoding equals the harness's two-model encoding of the same plan exactly.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from turboquant_pro import packed_weights as PW  # noqa: E402
from turboquant_pro import weight_codec as C  # noqa: E402
from turboquant_pro import weight_plan as W  # noqa: E402
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


@pytest.mark.parametrize("damp", [0.0, 0.01])
def test_codes_and_grid_are_exactly_the_weights_the_codec_returns(damp):
    """``codes=True`` changes no output, and ``codes * step + lo`` per group is the
    returned weights bit for bit: the codes a TQPW file stores are the codec's."""
    g = torch.Generator().manual_seed(4)
    ws = [torch.randn(r, 384, generator=g) for r in (8, 16, 24)]
    S, widths = _correlated(384), [2, 5, 8]

    def rebuilt(r, lo, st):
        out, inp = r.shape
        x = r.float().reshape(out, inp // C.GROUP, C.GROUP)
        return (x * st[..., None] + lo[..., None]).reshape(out, inp)

    plain = C.gptq_stack(ws, S, widths, damp=damp)
    for (q, r, lo, st), p, b in zip(
        C.gptq_stack(ws, S, widths, damp=damp, codes=True), plain, widths
    ):
        assert torch.equal(q, p)
        assert int(r.max()) <= 2**b - 1 and lo.shape == (q.shape[0], 3)
        assert torch.equal(rebuilt(r, lo, st), q)
    for b in widths:
        q, r, lo, st = C.rtn(ws[0], b, codes=True)
        assert torch.equal(q, C.rtn(ws[0], b))
        assert torch.equal(rebuilt(r, lo, st), q)


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
    summary = C.encode_model(model, plan, calib, codec)
    got = C.linear_modules(model)
    assert list(got) == list(vm)
    for n in vm:
        assert torch.equal(got[n].weight, vm[n].weight), n
    assert any(not torch.equal(got[n].weight, rm[n].weight) for n in rm)
    # The stored form: one matrix each, in model order, at the planned width and
    # the plan's stored bits; float16 grid rounding is measured, and small.
    packed = summary["packed"]
    assert [m.name for m in packed] == list(vm)
    for m in packed:
        assert m.bits == plan["bits"][m.name]
        assert m.payload_bits == W.stored_bits(vm[m.name].weight.numel(), m.bits)
    assert 0 <= summary["grid_rounding"]["max_steps"] < 0.25


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
    assert not man["model_saved"] and not (out / "config.json").exists()

    # The file holds exactly the codes and grid encode_model produces, at the
    # plan's stored bits.
    want = R.load(str(tiny), "cpu")
    summary = C.encode_model(
        want, plan, C.calibration_windows(_Tok(), TEXT, 3, 64), "gptq"
    )
    meta, stored = PW.read(str(out / "weights.tqpw"))
    assert meta["codec"] == "gptq" and meta["plan_sha256"] == man["plan_sha256"]
    for a, b in zip(summary["packed"], stored):
        assert (a.name, a.bits, a.shape) == (b.name, b.bits, b.shape)
        assert np.array_equal(a.codes, b.codes)
        assert np.array_equal(a.grid.view(np.uint16), b.grid.view(np.uint16))
    mods = C.linear_modules(want)
    plan_bits = sum(
        W.stored_bits(mods[n].weight.numel(), plan["bits"][n]) for n in mods
    )
    assert man["packed"]["payload_bits"] == plan_bits
    assert man["packed"]["bytes"] == os.path.getsize(out / "weights.tqpw")

    # What is stored is what runs: --save-model writes the decoded file, and
    # decode-weights rebuilds the same model from the base and the file alone.
    saved, rebuilt = tmp_path / "saved", tmp_path / "rebuilt"
    assert cli_main(argv + ["--out", str(saved), "--save-model"]) == 0  # last wins
    dec = ["plan", "decode-weights", "--packed", str(out / "weights.tqpw")]
    assert cli_main(dec + ["--model-path", str(tiny), "--out", str(rebuilt)]) == 0
    C.apply_packed(want, stored)
    for d in (saved, rebuilt):
        gm = C.linear_modules(R.load(str(d), "cpu"))
        assert all(torch.equal(mods[n].weight, gm[n].weight) for n in mods)

    assert cli_main(argv + ["--n-calib", "999"]) == 2  # too little text, refused
    other = dict(plan, stored_bits=plan_bits + 8)  # a plan for another model
    pp.write_text(json.dumps(other))
    assert cli_main(argv) == 2
    bad = tmp_path / "bad.tqpw"
    blob = bytearray((out / "weights.tqpw").read_bytes())
    blob[-1] ^= 0xFF
    bad.write_bytes(bytes(blob))
    dec = ["plan", "decode-weights", "--packed", str(bad), "--model-path", str(tiny)]
    assert cli_main(dec + ["--out", str(tmp_path / "x")]) == 2  # CRC refuses it


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
    argv += ["--device", "cpu", "--out", str(enc), "--save-model"]
    assert cli_main(argv) == 0
    before = C.linear_modules(R.load(str(tiny), "cpu"))
    after = C.linear_modules(R.load(str(enc), "cpu"))
    assert all(not torch.equal(before[n].weight, after[n].weight) for n in before)
    man = json.loads((enc / "weight_encoding.json").read_text())
    assert man["cost_table_hash"] == doc["cost_table_hash"]
    # The file is the plan's byte count: the payload is exactly its stored bits.
    assert man["packed"]["payload_bits"] == doc["stored_bits"]
