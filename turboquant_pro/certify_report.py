# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Certificate verify / report logic behind ``tqp certify`` and ``tqp verify``.

Everything here is independent of argparse: the functions take a certificate
document (the ``turboquant-pro/rank-certificate`` JSON emitted by
``tqp certify`` -- see ``docs/CERTIFICATE_SPEC.md`` and
``schemas/rank_certificate.schema.json``) plus plain values, and return
problems, sections, or rendered text. ``cli.py`` owns the argument parsing and
the exit codes; this module owns *what* is checked and reported:

* :func:`_verify_schema` -- structural + self-consistency checks on a document
  (no input data needed).
* :func:`_verify_recompute` -- independent reproduction: recompute the rank
  certificate from ``--original`` / ``--reconstructed`` and compare hashes
  and floors against what was recorded.
* :func:`_certify_reference` / :func:`_certify_environment` -- the optional
  ``reference`` (consumer-relative distortion against a registered
  read-operator provider) and ``environment`` (provenance) sections.
* :func:`_certify_html` -- the self-contained HTML report of a certificate.
* :func:`_certify_summary` / :func:`_verify_summary` -- the text summaries
  printed by the CLI.

The certificate *mathematics* (kappa, mu_hat, Kendall/Spearman floors) lives
in :mod:`turboquant_pro.rank_certificate`; this module only checks, decorates
and renders the resulting documents. Names keep their leading underscore
because ``cli.py`` re-exports them unchanged for existing callers.
"""

from __future__ import annotations

import argparse


def _sha256_array(arr) -> str:
    import hashlib

    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _certify_summary(doc: dict) -> str:
    c = doc["certificate"]
    lines = [
        f"# tqp certify  schema={doc['schema']} v{doc['schema_version']}  "
        f"tool={doc['tool_version']}",
        f"metric={doc['params']['metric']}  anchors={doc['params']['n_anchors']}  "
        f"pairs={c['n_pairs']}",
        f"  kappa (robust distortion) = {c['kappa']:.4f}",
        f"  mu_hat (concentration)    = {c['mu_hat']:.4f}",
        f"  Kendall  tau  floor       >= {c['tau_floor']:.4f}",
        f"  Spearman rho  floor       >= {c['spearman_floor']:.4f}",
        f"  max certifiable kappa     = {c['max_certifiable_kappa']:.4f}",
        f"=> {doc['interpretation']}",
    ]
    return "\n".join(lines)


def _certify_reference(args, orig, recon) -> dict:
    """Build the certificate's ``reference`` section.

    The point of the section is that a consumer-relative number is not
    interpretable on its own. Two defensible references for a single
    attention head differ by about 0.3 in subspace overlap, so the provider
    identity and a hash of the operator itself travel with the figure.
    """
    import hashlib
    import json as _json

    import numpy as _np

    from .read_operators import (
        consumer_distortion,
        create_read_operator,
        error_covariance,
        get_read_operator,
    )

    config = {}
    if getattr(args, "reference_config", None):
        config = _json.loads(args.reference_config)
        if not isinstance(config, dict):
            raise ValueError("--reference-config must be a JSON object")

    spec = get_read_operator(args.reference)
    provider = create_read_operator(args.reference, **config)

    a = _np.asarray(orig, dtype=_np.float64)
    b = _np.asarray(recon, dtype=_np.float64)
    a2 = a.reshape(-1, a.shape[-1])
    P = _np.asarray(provider.operator(a2, **config), dtype=_np.float64)
    sigma = error_covariance(a, b)

    trace = float(_np.trace(P))
    s2 = float((P**2).sum())
    eff = (trace * trace / s2) if s2 > 0 else None

    return {
        "provider": spec.name,
        "exact": bool(spec.exact),
        "description": spec.description,
        "dim": int(P.shape[0]),
        "operator_sha256": hashlib.sha256(
            _np.ascontiguousarray(P).tobytes()
        ).hexdigest(),
        "trace": trace,
        "effective_rank": eff,
        "consumer_distortion": consumer_distortion(P, sigma),
        "reconstruction_distortion": float(_np.trace(sigma)),
        "config": config,
    }


def _certify_environment() -> dict:
    """Software/hardware provenance for a certificate's ``environment`` section."""
    import platform
    import subprocess

    import numpy

    from turboquant_pro import __version__

    def _git() -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:  # noqa: BLE001
            return None

    return {
        "tool_version": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "git_commit": _git(),
        "hardware": platform.processor() or None,
    }


def _certify_html(doc: dict) -> str:
    """A small self-contained HTML report of a certificate document."""
    import html

    cert = doc.get("certificate", {})
    passed = doc.get("passed")
    status = "PASS" if passed else "REVIEW"
    color = "#2f855a" if passed else "#b7791f"
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in cert.items()
    )
    extra = ""
    for key in ("task", "environment"):
        if key in doc:
            body = "".join(
                f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
                for k, v in doc[key].items()
            )
            extra += f"<h2>{key}</h2><table>{body}</table>"
    if "limitations" in doc:
        items = "".join(f"<li>{html.escape(x)}</li>" for x in doc["limitations"])
        extra += f"<h2>limitations</h2><ul>{items}</ul>"
    interp = html.escape(str(doc.get("interpretation", "")))
    tool = html.escape(str(doc.get("tool_version", "")))
    when = html.escape(str(doc.get("created_utc", "")))
    schema = html.escape(str(doc.get("schema", "")))
    return (
        "<!doctype html><meta charset=utf-8>"
        f"<title>certificate — {schema}</title>"
        "<style>body{font:15px/1.6 system-ui,sans-serif;max-width:760px;margin:40px "
        "auto;padding:0 20px;color:#1a2222}h1{font-size:22px;margin:0 0 4px}"
        "table{border-collapse:collapse;width:100%;margin:8px 0 20px}"
        "td{border-bottom:1px solid #e2e8e8;padding:6px 8px;font-family:"
        "ui-monospace,monospace;font-size:13px}td:first-child{color:#5c6b6d}"
        "h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#5c6b6d;"
        "margin:22px 0 6px}.badge{display:inline-block;padding:4px 12px;border-radius:"
        "999px;color:#fff;font-weight:600;font-size:13px;font-family:ui-monospace}"
        ".interp{color:#3a4a4a}</style>"
        "<h1>TurboQuant Pro certificate</h1>"
        f"<p><span class=badge style='background:{color}'>{status}</span> "
        f"<span class=interp>{interp}</span></p>"
        f"<p style='color:#5c6b6d;font-size:13px'>tool {tool} · {when}</p>"
        f"<h2>certificate</h2><table>{rows}</table>"
        f"{extra}"
        "<p style='color:#5c6b6d;font-size:12px'>Acceptance is rank fidelity, never "
        "reconstruction cosine. Full spec: docs/CERTIFICATE_SPEC.md.</p>"
    )


def _verify_schema(doc: dict) -> list[str]:
    """Structural + self-consistency checks on a certificate doc (no data needed).

    Returns a list of human-readable problems; empty means the certificate is
    well-formed and internally sane (recognized schema, required fields present,
    rank statistics inside their valid ranges)."""
    import math

    problems: list[str] = []
    schema = doc.get("schema")
    if schema not in (
        "turboquant-pro/rank-certificate",
        "turboquant-pro/index-certificate",
    ):
        problems.append(f"unrecognized schema {schema!r}")
    if doc.get("schema_version") != 1:
        problems.append(f"unsupported schema_version {doc.get('schema_version')!r}")
    for key in ("tool_version", "params", "certificate", "interpretation", "passed"):
        if key not in doc:
            problems.append(f"missing top-level key {key!r}")
    if not isinstance(doc.get("passed"), bool):
        problems.append("`passed` is not a boolean")

    def _num(x) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(x)

    cert = doc.get("certificate") or {}
    for key in ("kappa", "mu_hat", "tau_floor", "spearman_floor", "n_pairs"):
        if key not in cert:
            problems.append(f"certificate missing {key!r}")
    for key in ("tau_floor", "spearman_floor"):
        v = cert.get(key)
        if v is not None and not (_num(v) and -1.0 - 1e-9 <= v <= 1.0 + 1e-9):
            problems.append(f"certificate.{key}={v!r} outside [-1, 1]")
    if "kappa" in cert and not (_num(cert["kappa"]) and cert["kappa"] >= -1e-9):
        problems.append(f"certificate.kappa={cert['kappa']!r} negative or non-finite")
    npairs = cert.get("n_pairs")
    if npairs is not None and not (isinstance(npairs, int) and npairs > 0):
        problems.append(f"certificate.n_pairs={npairs!r} is not a positive int")
    return problems


def _verify_recompute(doc: dict, args: argparse.Namespace) -> dict:
    """Recompute the rank certificate from ``--original``/``--reconstructed`` and
    compare to the recorded hashes + floors. Independent reproduction."""
    import numpy as np

    from turboquant_pro import certificate_from_embeddings

    if doc.get("schema") != "turboquant-pro/rank-certificate":
        return {
            "skipped": True,
            "error": "recompute (--original/--reconstructed) is only defined for "
            f"rank certificates; this is {doc.get('schema')!r}",
        }
    try:
        orig = np.asarray(np.load(args.original))
        recon = np.asarray(np.load(args.reconstructed))
    except Exception as e:  # noqa: BLE001
        return {"skipped": True, "error": f"cannot load inputs: {e}"}
    if orig.ndim > 2:
        orig = orig.reshape(-1, orig.shape[-1])
        recon = recon.reshape(-1, recon.shape[-1])
    if orig.shape != recon.shape or orig.ndim != 2:
        return {"skipped": True, "error": f"bad/mismatched input shapes {orig.shape}"}

    inp = doc.get("inputs", {})
    hashes = {
        "original": (
            _sha256_array(orig),
            (inp.get("original") or {}).get("sha256"),
        ),
        "reconstructed": (
            _sha256_array(recon),
            (inp.get("reconstructed") or {}).get("sha256"),
        ),
    }
    hash_match = {k: (h == rec) for k, (h, rec) in hashes.items()}
    hashes_ok = all(rec is not None and h == rec for h, rec in hashes.values())

    p = doc.get("params", {})
    cert = certificate_from_embeddings(
        orig,
        recon,
        n_anchors=int(p.get("n_anchors", 200)),
        metric=p.get("metric", "cosine"),
        seed=int(p.get("seed", 0)),
    )
    cvals = cert.as_dict()
    recorded = doc.get("certificate", {})
    fields = ("kappa", "mu_hat", "tau_floor", "spearman_floor")
    deltas, values_ok = {}, True
    for k in fields:
        rv, gv = recorded.get(k), cvals.get(k)
        if rv is None or gv is None:
            continue
        d = abs(float(gv) - float(rv))
        deltas[k] = d
        if d > args.atol + args.rtol * abs(float(rv)):
            values_ok = False
    return {
        "skipped": False,
        "hash_match": hash_match,
        "hashes_ok": hashes_ok,
        "recomputed": {k: cvals.get(k) for k in fields},
        "recorded": {k: recorded.get(k) for k in fields},
        "abs_delta": deltas,
        "tol": {"atol": args.atol, "rtol": args.rtol},
        "match": bool(hashes_ok and values_ok),
    }


def _verify_summary(r: dict) -> str:
    c = r["checks"]
    cert = r["certificate"]
    lines = [
        f"# tqp verify  {cert['path']}  (schema={cert['schema']})",
        f"  schema/self-consistency: {'OK' if c['schema_ok'] else 'FAILED'}",
    ]
    lines += [f"    - {p}" for p in c.get("schema_problems", [])]
    rc = c.get("recompute")
    if rc is not None:
        if rc.get("skipped"):
            lines.append(f"  recompute: skipped — {rc.get('error')}")
        else:
            lines.append(
                f"  recompute vs recorded: "
                f"{'MATCH' if rc['match'] else 'MISMATCH'} "
                f"(input hashes {'ok' if rc['hashes_ok'] else 'DIFFER'})"
            )
            for k, d in rc.get("abs_delta", {}).items():
                lines.append(f"    {k}: |Δ|={d:.2e}")
    lines.append(f"=> {'VERIFIED' if r['verified'] else 'NOT VERIFIED'}")
    return "\n".join(lines)
