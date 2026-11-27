"""`tqp monitor --observer` / `--certificate` (issue #173, phase 2).

The monitor measures cosine between original and reconstructed vectors. A
contract's floor is on the consumer's own metric (for example recall@10), so the
monitor reports the declared floor beside its measurement and says it did not
enforce it. What it does check against a contract is the certificate: the
contract it was issued for, and the coverage of the current window.
"""

from __future__ import annotations

import json

import numpy as np

from turboquant_pro.cli import main
from turboquant_pro.observer import ObserverContract, save_contract

D = 16


def _contract(**over):
    d = {
        "schema": "turboquant-pro/observer-contract",
        "profile": "tqp-observer/1",
        "observer": "svc",
        "target": "embedding",
        "consumers": [{"metric": "read_operator", "config": {"provider": "identity"}}],
        "requirements": {
            "floor": {"metric": "recall@10", "minimum": 0.995, "confidence": 0.95},
            "worst_stratum_minimum": 0.98,
        },
    }
    d.update(over)
    return ObserverContract.from_dict(d)


def _pair(tmp_path, n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, D)).astype(np.float32)
    y = (x + 0.01 * rng.standard_normal((n, D))).astype(np.float32)
    po, pr = tmp_path / "o.npy", tmp_path / "r.npy"
    np.save(po, x)
    np.save(pr, y)
    return str(po), str(pr), x


def _certify(tmp_path, po, pr, contract_path):
    cert = tmp_path / "cert.json"
    main(
        [
            "certify",
            "--original",
            po,
            "--reconstructed",
            pr,
            "--anchors",
            "16",
            "--observer",
            contract_path,
            "--validity",
            "--out",
            str(cert),
        ]
    )
    return str(cert)


def _monitor(po, pr, capsys, *extra):
    rc = main(["monitor", "--original", po, "--reconstructed", pr, *extra])
    return rc, capsys.readouterr()


def test_monitor_without_observer_is_unchanged(tmp_path, capsys):
    po, pr, _ = _pair(tmp_path)
    rc, out = _monitor(po, pr, capsys)
    assert rc == 0
    doc = json.loads(out.out)
    assert "observer" not in doc and "turboquant_certificate_valid" not in doc


def test_observer_section_names_the_contract_and_declines_to_enforce(tmp_path, capsys):
    po, pr, _ = _pair(tmp_path)
    c = _contract()
    cpath = str(tmp_path / "svc.tqo")
    save_contract(c, cpath)
    rc, out = _monitor(po, pr, capsys, "--observer", cpath, "--format", "json")
    assert rc == 0
    obs = json.loads(out.out)["observer"]
    assert obs["observer"] == "svc" and obs["sha256"] == c.digest()
    assert obs["declared_floor"] == {
        "metric": "recall@10",
        "minimum": 0.995,
        "confidence": 0.95,
    }
    assert obs["declared_worst_stratum_minimum"] == 0.98
    # the monitor measures cosine, not recall@10: it must say so, not pass or fail it
    assert obs["floor_enforced"] is False
    assert "cosine" in obs["floor_note"] and "recall@10" in obs["floor_note"]


def test_invalid_contract_is_refused_with_exit_2(tmp_path, capsys):
    po, pr, _ = _pair(tmp_path)
    bad = _contract(consumers=[{"metric": "no_such_consumer", "config": {}}])
    cpath = str(tmp_path / "bad.tqo")
    save_contract(bad, cpath)
    rc, out = _monitor(po, pr, capsys, "--observer", cpath)
    assert rc == 2
    assert "no_such_consumer" in out.err
    assert out.out == ""  # nothing is reported against a contract that is not valid


def test_missing_contract_file_is_exit_2(tmp_path, capsys):
    po, pr, _ = _pair(tmp_path)
    rc, out = _monitor(po, pr, capsys, "--observer", str(tmp_path / "absent.tqo"))
    assert rc == 2 and "cannot read contract" in out.err and out.out == ""


def test_certificate_for_this_contract_is_valid_on_the_same_data(tmp_path, capsys):
    po, pr, _ = _pair(tmp_path)
    cpath = str(tmp_path / "svc.tqo")
    save_contract(_contract(), cpath)
    cert = _certify(tmp_path, po, pr, cpath)
    capsys.readouterr()
    rc, out = _monitor(
        po, pr, capsys, "--observer", cpath, "--certificate", cert, "--format", "json"
    )
    doc = json.loads(out.out)
    assert rc == 0, out
    assert doc["certificate"]["status"] == "VALID"
    assert doc["turboquant_certificate_valid"] == 1


def test_certificate_issued_for_another_contract_is_stale_exit_1(tmp_path, capsys):
    po, pr, _ = _pair(tmp_path)
    cpath = str(tmp_path / "svc.tqo")
    save_contract(_contract(), cpath)
    cert = _certify(tmp_path, po, pr, cpath)
    capsys.readouterr()
    other = str(tmp_path / "other.tqo")
    save_contract(_contract(observer="other"), other)
    rc, out = _monitor(
        po, pr, capsys, "--observer", other, "--certificate", cert, "--format", "json"
    )
    doc = json.loads(out.out)
    assert rc == 1
    assert doc["certificate"]["status"] == "STALE"
    assert "observer contract changed" in doc["certificate"]["reason"]
    assert doc["certificate"]["action"] == "REPLAN"


def test_shifted_data_makes_the_certificate_stale(tmp_path, capsys):
    po, pr, x = _pair(tmp_path)
    cpath = str(tmp_path / "svc.tqo")
    save_contract(_contract(), cpath)
    cert = _certify(tmp_path, po, pr, cpath)
    capsys.readouterr()
    np.save(po, x + 4.0)
    np.save(pr, x + 4.0)
    rc, out = _monitor(
        po, pr, capsys, "--observer", cpath, "--certificate", cert, "--format", "json"
    )
    doc = json.loads(out.out)
    assert rc == 1 and doc["certificate"]["status"] == "STALE"
    assert doc["certificate"]["action"] == "RECERTIFY"


def test_certificate_without_a_validity_section_is_refused(tmp_path, capsys):
    po, pr, _ = _pair(tmp_path)
    cert = tmp_path / "plain.json"
    main(
        [
            "certify",
            "--original",
            po,
            "--reconstructed",
            pr,
            "--anchors",
            "16",
            "--out",
            str(cert),
        ]
    )
    capsys.readouterr()
    rc, out = _monitor(po, pr, capsys, "--certificate", str(cert))
    assert rc == 2 and "validity" in out.err


def test_text_format_shows_the_observer_and_certificate(tmp_path, capsys):
    po, pr, _ = _pair(tmp_path)
    cpath = str(tmp_path / "svc.tqo")
    save_contract(_contract(), cpath)
    cert = _certify(tmp_path, po, pr, cpath)
    capsys.readouterr()
    rc, out = _monitor(
        po, pr, capsys, "--observer", cpath, "--certificate", cert, "--format", "text"
    )
    assert rc == 0
    assert "observer" in out.out and "svc" in out.out
    assert "certificate" in out.out and "VALID" in out.out
