"""``tqp`` — the unified turboquant-pro CLI.

One command over the whole toolkit, following the pipeline
``trace -> plan -> compress -> certify -> replay -> monitor``:

    tqp version
    tqp plugin list                 # registered quantizer plugins
    tqp plugin conformance [names]  # run the container-contract conformance kit
    tqp trace <hf-model>            # operator regime -> (A2) discipline per tensor
    tqp probe --npy keys.npy        # (A2) consumer-metric quantizer-family probe
    tqp plan embeddings|kv ...      # recipe + rank-certificate / risk preview
    tqp certify --original o.npy --reconstructed r.npy   # rank certificate.json
    tqp replay <claim|all>          # execute claim reproductions from claims.yaml
    tqp monitor --original o.npy --reconstructed r.npy   # production metrics
    tqp anatomy --npy x.npy         # hub anatomy vector (skew alone: non-identifying)
    tqp hubdiff --original o.npy --reconstructed r.npy   # anti-hub/hub-rank oracle

Coherence rule across every command: the acceptance signal is **rank fidelity /
the (A2) consumer metric / a distribution-free certificate** — never
reconstruction cosine on its own. Cosine appears only as a labelled secondary
diagnostic, and where it is the base signal (``monitor``) it is guarded by the
(A2) tangential-fraction / radial-drift statistics, because cosine can read
fine while the ranking the consumer actually uses collapses
(``docs/KV_KEYS_FINDING.md``).

Registered as the ``tqp`` console script; ``turboquant-pro`` remains the
AutoConfig entry point.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter


# ------------------------------------------------------------------ commands
def _cmd_version(args: argparse.Namespace) -> int:
    from turboquant_pro import __version__

    print(f"turboquant-pro {__version__}")
    return 0


def _cmd_plugin_list(args: argparse.Namespace) -> int:
    from turboquant_pro import available_plugins

    specs = available_plugins(target=args.target)
    if not specs:
        extra = f" for target {args.target!r}" if args.target else ""
        print(f"no quantizer plugins registered{extra}")
        return 0
    name_w = max(len(n) for n in specs)
    tier_w = max(len(s.tier) for s in specs.values())
    print(f"{'NAME':<{name_w}}  {'TIER':<{tier_w}}  TARGETS")
    for name, s in specs.items():
        print(f"{name:<{name_w}}  {s.tier:<{tier_w}}  {', '.join(sorted(s.targets))}")
        if args.verbose and s.description:
            print(f"{'':<{name_w}}    {s.description}")
    return 0


def _cmd_plugin_conformance(args: argparse.Namespace) -> int:
    import numpy as np

    from turboquant_pro import available_plugins, create_quantizer, run_conformance

    names = args.names or list(available_plugins(target=args.target))
    if not names:
        print("no plugins to check")
        return 0
    rng = np.random.default_rng(0)
    if args.shape:
        x = rng.standard_normal(tuple(int(d) for d in args.shape.split(","))).astype(
            np.float32
        )
        kv_config: dict = {}
    else:
        # Canonical KV block (B, H, S, D) with a per-head DC offset — the shape the
        # in-tree KV plugins (per_channel keys, polar values) expect, and the DC
        # offset per_channel's whole point. Override with --shape for other plugins.
        h, s, d = args.heads, args.seq, args.dim
        off = rng.uniform(-4.0, 4.0, size=(1, h, 1, d))
        x = (off + rng.standard_normal((1, h, s, d))).astype(np.float32)
        # KV quantizers size their internal rotation/grids to (head_dim, n_heads);
        # build them to match the sample block (ignored by plugins that don't take it).
        kv_config = {"head_dim": d, "n_heads": h}
    any_fail = False
    for name in names:
        print(f"== {name} ==")
        try:
            try:
                q = create_quantizer(name, **kv_config)
            except TypeError:
                q = create_quantizer(name)  # plugin doesn't take head_dim/n_heads
            report = run_conformance(q, x)
        except Exception as e:  # noqa: BLE001 - report, don't crash the whole run
            any_fail = True
            print(f"  ERROR instantiating/running: {type(e).__name__}: {e}")
            continue
        print(report)
        any_fail = any_fail or not report.passed
    return 1 if any_fail else 0


def _cmd_trace(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoConfig, AutoModel
    except ImportError:
        print(
            "tqp trace needs PyTorch + transformers:\n"
            "  pip install 'turboquant-pro[torch]' transformers",
            file=sys.stderr,
        )
        return 2

    from turboquant_pro import QuantTarget, trace_operators

    try:
        target = QuantTarget(args.target)
    except ValueError:
        print(
            f"unknown --target {args.target!r}; choose from "
            f"{[t.value for t in QuantTarget]}",
            file=sys.stderr,
        )
        return 2

    # Build the architecture on the meta device: real module structure and names,
    # zero materialized weights — so tracing a 7B model costs no download/RAM.
    cfg = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    with torch.device("meta"):
        model = AutoModel.from_config(cfg, trust_remote_code=args.trust_remote_code)

    plan = trace_operators(model, prefer=args.prefer)
    tensors = plan.tensors
    if not tensors:
        print("no quantizable tensors found")
        return 0

    print(
        f"# {args.model}  (target={target.value}, prefer={args.prefer}, "
        f"fx-traced={plan.traced}, tensors={len(tensors)})"
    )
    by_regime: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    for name, t in tensors.items():
        disc = t.discipline(target)
        by_regime[t.regime.value] += 1
        by_family[disc.family] += 1
        if args.verbose:
            print(
                f"{name}\n"
                f"    regime={t.regime.value} (conf {t.confidence:.2f}) "
                f"-> family={disc.family} protect_dc={disc.protect_dc} "
                f"sens={disc.sensitivity}"
            )
    print("\nregime distribution:")
    for r, n in by_regime.most_common():
        print(f"  {r:<16} {n}")
    print("(A2) discipline family distribution:")
    for f, n in by_family.most_common():
        print(f"  {f:<16} {n}")
    if not args.verbose:
        print("\n(pass --verbose for the per-tensor table)")
    return 0


_PROBE_CONSUMERS = ("cosine", "l2", "attention_logits")


def _demo_probe_batch(kind: str, seed: int):
    """A labeled synthetic batch for `tqp probe --demo` (illustration, not data)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n, d = 512, 64
    if kind == "isotropic":
        # Directions spread over the sphere: both families track the consumer.
        return rng.standard_normal((n, d)).astype(np.float32)
    # "dc_offset": the v1.2.0 KV-keys regime — a large shared per-channel offset
    # with tiny informative directions riding on top. Displacements read fully
    # tangential, yet per-vector polar quantization rounds the informative angle
    # away (its cell swamps the signal) while per-channel affine keeps it.
    offset = rng.uniform(-6.0, 6.0, size=(1, d))
    signal = 0.15 * rng.standard_normal((n, d))
    return (offset + signal).astype(np.float32)


def _cmd_probe(args: argparse.Namespace) -> int:
    import json

    import numpy as np

    from turboquant_pro import probe_quotient

    if args.demo:
        batch = _demo_probe_batch(args.demo, args.seed)
        source = f"demo:{args.demo} (SYNTHETIC illustration, not measured data)"
    elif args.npy:
        try:
            batch = np.asarray(np.load(args.npy))
        except Exception as e:  # noqa: BLE001
            print(f"probe: cannot load {args.npy!r}: {e}", file=sys.stderr)
            return 2
        source = args.npy
    else:
        print(
            "tqp probe needs input: --npy PATH or --demo {isotropic,dc_offset}",
            file=sys.stderr,
        )
        return 2

    note = ""
    if batch.ndim > 2:
        flat = batch.reshape(-1, batch.shape[-1])
        note = f" [reshaped {batch.shape} -> {flat.shape}, rows = last-axis vectors]"
        batch = flat

    queries = None
    if args.queries:
        try:
            queries = np.asarray(np.load(args.queries))
        except Exception as e:  # noqa: BLE001
            print(f"probe: cannot load {args.queries!r}: {e}", file=sys.stderr)
            return 2
        if queries.ndim > 2:
            queries = queries.reshape(-1, queries.shape[-1])

    try:
        result = probe_quotient(
            batch,
            consumer=args.consumer,
            queries=queries,
            bits=args.bits,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"probe: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    print(f"# tqp probe  source={source}{note}")
    print(f"consumer={result.consumer}  bits={args.bits}")
    print(f"  spearman(polar)        = {result.spearman_polar:.4f}")
    print(f"  spearman(per_channel)  = {result.spearman_per_channel:.4f}")
    print(f"  margin                 = {result.margin:.4f}")
    print(f"  median tangential frac = {result.median_tangential_fraction:.4f}")
    print(f"  median unit displace   = {result.median_unit_displacement:.4f}")
    print(f"=> recommend: {result.recommendation}")
    print(
        "(calibration-time family selection; validate the shipped path "
        "end-to-end. Probe guards direction-concentration failures; pair with "
        "`tqp monitor` for radial drift in production.)"
    )
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    import json

    import numpy as np

    from turboquant_pro import QualityMonitor

    try:
        orig = np.asarray(np.load(args.original))
        recon = np.asarray(np.load(args.reconstructed))
    except Exception as e:  # noqa: BLE001
        print(f"monitor: cannot load inputs: {e}", file=sys.stderr)
        return 2
    if orig.shape != recon.shape:
        print(
            f"monitor: shape mismatch — original {orig.shape} != "
            f"reconstructed {recon.shape}",
            file=sys.stderr,
        )
        return 2
    if orig.ndim == 1:
        orig, recon = orig[None, :], recon[None, :]
    elif orig.ndim > 2:
        orig = orig.reshape(-1, orig.shape[-1])
        recon = recon.reshape(-1, recon.shape[-1])

    mon = QualityMonitor(
        quality_floor=args.floor,
        window_size=args.window,
        tangential_floor=args.tangential_floor,
    )
    mon.record_batch(orig.astype(np.float64), recon.astype(np.float64))
    metrics = mon.metrics_dict()
    stats = mon.stats()

    if args.format == "json":
        print(json.dumps(metrics, indent=2))
    elif args.format == "prometheus":
        for k, v in metrics.items():
            print(f"# TYPE {k} gauge")
            print(f"{k} {v}")
    else:  # text
        for k, v in stats.items():
            print(f"{k:34} {v}")
    # Exit non-zero when unhealthy — cosine below floor OR (A2) tangential
    # collapse — so `tqp monitor` gates on the same coherent signal as is_healthy.
    return 0 if stats["is_healthy"] else 1


def _json_safe(obj):
    """Recursively replace non-finite floats with ``None`` so emitted JSON is
    spec-valid.

    Python's ``json`` emits bare ``NaN`` / ``Infinity`` by default — not valid
    JSON, and rejected by strict validators and many parsers. A non-finite
    measurement (e.g. a NaN distortion ``kappa`` on a degenerate corpus) is a
    real, meaningful outcome, so it is serialized as JSON ``null`` rather than
    dropped. ``np.float64`` is a ``float`` subclass and is covered; keys are left
    untouched.
    """
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _emit_doc(doc: dict, out: str | None, fmt: str, summary: str) -> bool:
    """Emit a result document: write to ``out`` (+summary), or print per ``fmt``.

    All emitted JSON is spec-valid: non-finite floats become ``null`` (see
    :func:`_json_safe`) and ``allow_nan=False`` is a hard guard against any that
    slip through, so a `tqp` JSON artifact never contains bare ``NaN``.

    Returns False only when an ``--out`` write fails (caller should exit 2).
    """
    import json

    doc = _json_safe(doc)
    if out:
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, allow_nan=False)
        except OSError as e:
            print(f"cannot write {out!r}: {e}", file=sys.stderr)
            return False
        print(f"wrote {out}")
        print(summary)
    elif fmt == "json":
        print(json.dumps(doc, indent=2, allow_nan=False))
    else:
        print(summary)
    return True


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


def _cmd_certify(args: argparse.Namespace) -> int:
    import numpy as np

    from turboquant_pro import __version__, certificate_from_embeddings

    try:
        orig = np.asarray(np.load(args.original))
        recon = np.asarray(np.load(args.reconstructed))
    except Exception as e:  # noqa: BLE001
        print(f"certify: cannot load inputs: {e}", file=sys.stderr)
        return 2
    orig_shape, recon_shape = orig.shape, recon.shape
    if orig.ndim > 2:
        orig = orig.reshape(-1, orig.shape[-1])
        recon = recon.reshape(-1, recon.shape[-1])
    if orig.shape != recon.shape:
        print(
            f"certify: shape mismatch — original {orig_shape} != "
            f"reconstructed {recon_shape}",
            file=sys.stderr,
        )
        return 2
    if orig.ndim != 2 or orig.shape[0] < 2:
        print(
            f"certify: need at least 2 vectors of shape (n, d); got {orig.shape}",
            file=sys.stderr,
        )
        return 2

    cert = certificate_from_embeddings(
        orig, recon, n_anchors=args.anchors, metric=args.metric, seed=args.seed
    )

    tau = cert.tau_floor
    if args.min_tau is not None:
        passed = bool(np.isfinite(tau) and tau >= args.min_tau)
        interp = (
            f"PASS: Kendall tau floor {tau:.4f} >= min-tau {args.min_tau}"
            if passed
            else f"FAIL: Kendall tau floor {tau:.4f} < min-tau {args.min_tau} "
            "— exact reranking required"
        )
    else:
        passed = not cert.vacuous
        interp = (
            f"certifies Kendall tau >= {tau:.4f}, Spearman rho >= "
            f"{cert.spearman_floor:.4f} (distribution-free)"
            if passed
            else "VACUOUS: no finite distortion certifies rank on this corpus "
            "— exact reranking required"
        )

    doc = {
        "schema": "turboquant-pro/rank-certificate",
        "schema_version": 1,
        "tool_version": __version__,
        "created_utc": _now_utc(),
        "inputs": {
            "original": {
                "path": args.original,
                "shape": list(orig_shape),
                "dtype": str(orig.dtype),
                "sha256": _sha256_array(orig),
            },
            "reconstructed": {
                "path": args.reconstructed,
                "shape": list(recon_shape),
                "dtype": str(recon.dtype),
                "sha256": _sha256_array(recon),
            },
        },
        "params": {"metric": args.metric, "n_anchors": args.anchors, "seed": args.seed},
        "certificate": cert.as_dict(),
        "interpretation": interp,
        "passed": passed,
    }

    # Optional richer envelope (additive; schema_version stays 1).
    if getattr(args, "task", None):
        doc["task"] = {"kind": args.task_kind, "target": args.task}
    if getattr(args, "environment", False):
        doc["environment"] = _certify_environment()
    if getattr(args, "limitation", None):
        doc["limitations"] = list(args.limitation)

    if getattr(args, "html", None):
        try:
            with open(args.html, "w", encoding="utf-8") as f:
                f.write(_certify_html(doc))
            print(f"wrote {args.html}")
        except OSError as e:
            print(f"cannot write {args.html!r}: {e}", file=sys.stderr)
            return 2

    if not _emit_doc(doc, args.out, args.format, _certify_summary(doc)):
        return 2
    return 0 if passed else 1


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


def _cmd_verify(args: argparse.Namespace) -> int:
    import json

    from turboquant_pro import __version__

    try:
        with open(args.certificate, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        print(f"verify: cannot read {args.certificate!r}: {e}", file=sys.stderr)
        return 2

    if bool(args.original) != bool(args.reconstructed):
        print(
            "verify: --original and --reconstructed must be given together to "
            "recompute; ignoring the lone one and checking schema only",
            file=sys.stderr,
        )

    problems = _verify_schema(doc)
    checks: dict = {"schema_ok": not problems, "schema_problems": problems}
    recompute = None
    if args.original and args.reconstructed:
        recompute = _verify_recompute(doc, args)
        checks["recompute"] = recompute

    match_ok = (
        recompute is None or recompute.get("skipped") or recompute.get("match") is True
    )
    verified = (not problems) and match_ok
    result = {
        "schema": "turboquant-pro/verification",
        "schema_version": 1,
        "tool_version": __version__,
        "created_utc": _now_utc(),
        "certificate": {
            "path": args.certificate,
            "schema": doc.get("schema"),
            "tool_version": doc.get("tool_version"),
        },
        "checks": checks,
        "verified": verified,
    }
    if not _emit_doc(result, args.out, args.format, _verify_summary(result)):
        return 2
    return 0 if verified else 1


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


def _now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _preview_certificate(emb, config: dict, seed: int):
    """Roundtrip an eval sample under ``config`` and return a rank certificate.

    The planner's acceptance signal is rank fidelity (a distribution-free
    Kendall/Spearman floor), not reconstruction cosine — cosine is the promise
    this project shows can read fine while ranking collapses.
    """
    import numpy as np

    from turboquant_pro import certificate_from_embeddings
    from turboquant_pro.pca import PCAMatryoshka
    from turboquant_pro.pgvector import TurboQuantPGVector

    dim = emb.shape[1]
    pca_dim = config.get("pca_dim")
    bits = config.get("bits")
    sample = emb[: min(600, len(emb))]
    if pca_dim is not None and pca_dim < dim:
        pca = PCAMatryoshka(input_dim=dim, output_dim=pca_dim)
        pca.fit(emb[:1000])
        if config.get("weighted"):
            pipe = pca.with_weighted_quantizer(
                avg_bits=config.get("avg_bits", 3.0), seed=seed
            )
        else:
            pipe = pca.with_quantizer(bits=bits, seed=seed)
        recon = np.stack([pipe.decompress(pipe.compress(v)) for v in sample])
    else:
        tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)
        recon = np.stack(
            [tq.decompress_embedding(tq.compress_embedding(v)) for v in sample]
        )
    return certificate_from_embeddings(sample, recon, metric="cosine", seed=seed)


def _plan_emb_summary(doc: dict) -> str:
    r = doc["recommended"]
    cp = doc["certificate_preview"]
    lines = [
        f"# tqp plan embeddings  target={doc['constraints']['target']!r}"
        + (
            f"  <= {doc['constraints']['max_bytes_per_vector']} B/vec"
            if doc["constraints"]["max_bytes_per_vector"] is not None
            else ""
        ),
        f"recommended: {r.get('label', '?')}  "
        f"ratio={r.get('ratio')}x  {r.get('bytes_per_vector')} B/vec",
    ]
    if "tau_floor" in cp:
        lines.append(
            f"  rank floor (cosine ranking): Kendall tau >= {cp['tau_floor']:.4f}, "
            f"Spearman rho >= {cp['spearman_floor']:.4f}"
            + ("  [VACUOUS — rerank required]" if cp.get("vacuous") else "")
        )
    else:
        lines.append(f"  rank certificate preview unavailable: {cp.get('error')}")
    lines.append(
        f"  [diagnostic only] reconstruction mean_cosine = "
        f"{r.get('mean_cosine'):.4f}"
    )
    lines.append(f"alternatives on the Pareto frontier: {len(doc['alternatives'])}")
    for flag in doc["risk_flags"]:
        lines.append(f"  ! {flag}")
    lines.append(f"reproduce: {doc['reproduction']}")
    return "\n".join(lines)


def _cmd_plan_embeddings(args: argparse.Namespace) -> int:
    import numpy as np

    from turboquant_pro import __version__, auto_compress

    try:
        emb = np.asarray(np.load(args.embeddings))
    except Exception as e:  # noqa: BLE001
        print(f"plan embeddings: cannot load {args.embeddings!r}: {e}", file=sys.stderr)
        return 2
    in_shape = emb.shape
    if emb.ndim > 2:
        emb = emb.reshape(-1, emb.shape[-1])
    if emb.ndim != 2 or emb.shape[0] < 2:
        print(
            f"plan embeddings: need (n, d) embeddings; got {in_shape}", file=sys.stderr
        )
        return 2
    dim = emb.shape[1]

    result = auto_compress(
        emb,
        target=args.target,
        sample_size=args.sample,
        seed=args.seed,
        verbose=False,
    )

    def _bpv(ratio: float) -> float:
        return round((dim * 4.0) / ratio, 2) if ratio else float("inf")

    frontier = [{**c, "bytes_per_vector": _bpv(c["ratio"])} for c in result.candidates]
    recommended = {**result.config, "bytes_per_vector": _bpv(result.ratio)}

    risk_flags: list[str] = []
    passed = True
    budget = args.max_bytes_per_vector
    if budget is not None:
        eligible = [c for c in frontier if c["bytes_per_vector"] <= budget]
        if eligible:
            recommended = max(eligible, key=lambda c: c["mean_cosine"])
        else:
            risk_flags.append(
                f"no recipe fits {budget} bytes/vector; showing the smallest"
            )
            recommended = min(frontier, key=lambda c: c["bytes_per_vector"])
            passed = False

    if len(frontier) < 2:
        risk_flags.append("thin Pareto frontier — few viable recipes at this target")

    # Acceptance signal = a distribution-free rank certificate, NOT cosine.
    try:
        cert = _preview_certificate(emb, recommended, args.seed)
        preview = {
            "metric": "cosine",
            "tau_floor": round(float(cert.tau_floor), 6),
            "spearman_floor": round(float(cert.spearman_floor), 6),
            "kappa": round(float(cert.kappa), 6),
            "vacuous": bool(cert.vacuous),
            "rerank_required": bool(cert.rerank_required),
        }
        if cert.vacuous:
            risk_flags.append(
                "rank certificate is vacuous — single-stage rank fidelity is not "
                "certifiable; exact reranking is required"
            )
            passed = False
    except Exception as e:  # noqa: BLE001 - preview must never crash the plan
        preview = {"error": f"{type(e).__name__}: {e}"}

    doc = {
        "schema": "turboquant-pro/embedding-plan",
        "schema_version": 1,
        "tool_version": __version__,
        "created_utc": _now_utc(),
        "input": {
            "path": args.embeddings,
            "shape": list(in_shape),
            "dtype": str(emb.dtype),
        },
        "constraints": {
            "target": args.target,
            "max_bytes_per_vector": budget,
        },
        "recommended": recommended,
        "certificate_preview": preview,
        "alternatives": frontier,
        "risk_flags": risk_flags,
        "note": (
            "auto_compress ranks the frontier on the target metric — a measured "
            "recall@k when the target is a recall target, else cosine/ratio. The "
            "plan's acceptance signal is the distribution-free rank-certificate "
            "preview; reconstruction cosine is reported only as a labelled "
            "diagnostic. Validate the final build with `tqp certify`."
        ),
        "reproduction": (
            "compress with the recommended recipe, then: "
            f"tqp certify --original {args.embeddings} "
            "--reconstructed <recon>.npy --metric cosine"
        ),
        "passed": passed,
    }
    if not _emit_doc(doc, args.out, args.format, _plan_emb_summary(doc)):
        return 2
    return 0 if passed else 1


def _plan_kv_summary(doc: dict) -> str:
    p = doc["policy"]
    lines = [
        f"# tqp plan kv  model={p['model']}  target={p['target']}",
        f"  {p['target_description']}",
        f"  keys  = {p['key_bits']}-bit   values = {p['value_bits']}-bit   "
        f"rope_aware={p['rope_aware']}",
        f"  head_dim={p['head_dim']}  n_kv_heads={p['n_kv_heads']}  "
        f"n_layers={p['n_layers']}  max_seq_len={p['max_seq_len']}",
        f"  est. KV cache = {p['estimated_kv_cache_gb']} GB  "
        f"({p['compression_ratio']}x, saves {p['saved_gb']} GB)",
    ]
    for flag in doc["risk_flags"]:
        lines.append(f"  ! {flag}")
    lines.append(f"reproduce: {doc['reproduction']}")
    return "\n".join(lines)


def _cmd_plan_kv(args: argparse.Namespace) -> int:
    from turboquant_pro import AutoConfig, __version__

    overrides = {}
    if args.context is not None:
        overrides["max_seq_len"] = args.context
    try:
        cfg = AutoConfig.from_pretrained(args.model, target=args.target, **overrides)
    except ValueError as e:
        print(f"plan kv: {e}", file=sys.stderr)
        return 2

    risk_flags: list[str] = []
    if cfg.key_bits < 4:
        risk_flags.append(
            f"keys below the 4-bit default (K{cfg.key_bits}) — the KV-keys finding: "
            "validate per-channel/asym key handling before trusting this"
        )
    if not cfg.rope_aware:
        risk_flags.append("RoPE-aware key protection is disabled")

    doc = {
        "schema": "turboquant-pro/kv-plan",
        "schema_version": 1,
        "tool_version": __version__,
        "created_utc": _now_utc(),
        "request": {
            "model": args.model,
            "target": args.target,
            "context": args.context,
        },
        "policy": cfg.summary(),
        "key_zero_point": cfg.key_zero_point,
        "risk_flags": risk_flags,
        "reproduction": (
            f"AutoConfig.from_pretrained({args.model!r}, target={args.target!r})"
            ".build_cache(...); or `tqp trace <hf-path>` for the operator view"
        ),
    }
    if not _emit_doc(doc, args.out, args.format, _plan_kv_summary(doc)):
        return 2
    return 0


def _replay_check_expected(claim: dict, cwd: str):
    """Load the claim's JSON outputs and check them against ``expected`` ranges.

    Keys ending ``_min``/``_max`` bound the like-named metric; a bare key
    requires exact equality. Returns ``(measured, checks)``.
    """
    import json
    import os

    results: dict = {}
    for out in claim.get("outputs", []) or []:
        if str(out).endswith(".json"):
            path = os.path.join(cwd, out)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    results.update(json.load(f))
    measured: dict = {}
    checks: dict = {}
    for key, bound in (claim.get("expected") or {}).items():
        if key.endswith("_min"):
            metric = key[:-4]
            ok = metric in results and results[metric] >= bound
        elif key.endswith("_max"):
            metric = key[:-4]
            ok = metric in results and results[metric] <= bound
        else:
            metric = key
            ok = results.get(metric) == bound
        measured[metric] = results.get(metric)
        checks[key] = bool(ok)
    return measured, checks


def _replay_summary(doc: dict) -> str:
    s = doc["summary"]
    lines = [
        f"# tqp replay  claims={s['n']}  reproduced={s['reproduced']}  "
        f"regressed={s['regressed']}  error={s['error']}  "
        f"manual={s['manual']}  dry_run={s['dry_run']}",
    ]
    for r in doc["claims"]:
        extra = ""
        if "duration_s" in r:
            extra = f"  ({r['duration_s']}s)"
        lines.append(f"  [{r['verdict']:>10}] {r['id']}{extra}")
    return "\n".join(lines)


def _cmd_replay(args: argparse.Namespace) -> int:
    import subprocess
    import time

    try:
        import yaml
    except ImportError:
        print(
            "tqp replay needs PyYAML:\n  pip install 'turboquant-pro[yaml]'",
            file=sys.stderr,
        )
        return 2

    try:
        with open(args.claims, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"replay: claims file not found: {args.claims}", file=sys.stderr)
        return 2
    except yaml.YAMLError as e:
        print(f"replay: cannot parse {args.claims}: {e}", file=sys.stderr)
        return 2

    claims = (data or {}).get("claims", {})
    if not claims:
        print(f"replay: no claims found in {args.claims}", file=sys.stderr)
        return 2

    if args.list:
        print(f"{'ID':<32} {'TRACK':<10} {'STATUS':<14} HARDWARE")
        for cid, c in claims.items():
            print(
                f"{cid:<32} {str(c.get('track', '?')):<10} "
                f"{str(c.get('status', '?')):<14} {c.get('hardware', '?')}"
            )
        return 0

    if args.target == "all":
        ids = [
            cid
            for cid, c in claims.items()
            if args.track is None or c.get("track") == args.track
        ]
    elif args.target is None:
        print(
            "replay: specify a claim id or 'all' (or --list). "
            "See `tqp replay --list`.",
            file=sys.stderr,
        )
        return 2
    elif args.target not in claims:
        print(
            f"replay: unknown claim {args.target!r}; see `tqp replay --list`",
            file=sys.stderr,
        )
        return 2
    else:
        ids = [args.target]
    if not ids:
        print("replay: no claims matched the selection", file=sys.stderr)
        return 2

    reports = []
    counts = {
        "reproduced": 0,
        "regressed": 0,
        "error": 0,
        "manual": 0,
        "dry_run": 0,
    }
    for cid in ids:
        c = claims[cid]
        cmd = c.get("full_command") if args.full else c.get("command")
        entry = {"id": cid, "track": c.get("track"), "status": c.get("status")}
        if not cmd:
            entry["verdict"] = "manual"
            entry["reference"] = c.get("reference", "")
            counts["manual"] += 1
            reports.append(entry)
            continue
        entry["command"] = cmd
        if args.dry_run:
            entry["verdict"] = "dry_run"
            counts["dry_run"] += 1
            reports.append(entry)
            continue
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                cwd=args.cwd,
            )
        except subprocess.TimeoutExpired:
            entry["verdict"] = "error"
            entry["drift_class"] = f"environment (timed out after {args.timeout}s)"
            counts["error"] += 1
            reports.append(entry)
            continue
        entry["duration_s"] = round(time.perf_counter() - t0, 3)
        entry["exit_code"] = proc.returncode
        if proc.returncode != 0:
            entry["verdict"] = "error"
            entry["drift_class"] = "environment or code (command exited non-zero)"
            entry["stderr_tail"] = proc.stderr.strip().splitlines()[-5:]
            counts["error"] += 1
            reports.append(entry)
            continue
        measured, checks = _replay_check_expected(c, args.cwd)
        entry["measured"] = measured
        entry["checks"] = checks
        if all(checks.values()):
            entry["verdict"] = "reproduced"
            counts["reproduced"] += 1
        else:
            entry["verdict"] = "regressed"
            entry["drift_class"] = "metric or data (results outside expected range)"
            counts["regressed"] += 1
        reports.append(entry)

    from turboquant_pro import __version__

    doc = {
        "schema": "turboquant-pro/replay-report",
        "schema_version": 1,
        "tool_version": __version__,
        "created_utc": _now_utc(),
        "claims_file": args.claims,
        "claims": reports,
        "summary": {"n": len(reports), **counts},
    }
    fmt = "text" if args.out is None and not args.json else "json"
    if not _emit_doc(doc, args.out, fmt, _replay_summary(doc)):
        return 2
    return 1 if (counts["regressed"] or counts["error"]) else 0


# ------------------------------------------------------------------ parser
def _load_ids(spec: str):
    """Ids from a path (one per line) or a literal comma-separated list."""
    import os

    import numpy as np

    if os.path.exists(spec):
        return np.loadtxt(spec, dtype=np.int64).reshape(-1)
    return np.array([int(t) for t in spec.split(",") if t.strip()], dtype=np.int64)


def _index_stats_summary(stats: dict) -> str:
    return (
        f"v{stats['format_version']} index: {stats['n_live']} live "
        f"/ {stats['n_rows']} rows ({stats['n_tombstoned']} tombstoned), "
        f"{stats['input_dim']}->{stats['output_dim']}d @ {stats['bits']}b, "
        f"~{stats['compression_ratio']}x, originals={stats['has_originals']}"
    )


def _cmd_index_create(args: argparse.Namespace) -> int:
    import numpy as np

    from .index import TQEIndex

    emb = np.asarray(np.load(args.embeddings))
    if emb.ndim != 2:
        print(f"index create: embeddings must be 2-D, got {emb.shape}", file=sys.stderr)
        return 2

    if getattr(args, "shard_size", 0):
        # Sharded: --out is a directory holding shard_*.tqe + manifest.json.
        from .sharded_index import ShardedIndex

        sh = ShardedIndex.create(
            emb,
            args.out,
            shard_size=args.shard_size,
            output_dim=args.output_dim,
            bits=args.bits,
            seed=args.seed,
            rotation=args.rotation,
            whiten=args.whiten,
            metric=args.metric,
            keep_originals=not args.no_originals,
        )
        print(f"wrote {args.out}/manifest.json ({sh.n_shards} shards)")
        print(
            f"sharded index: {sh.n_rows} rows across {sh.n_shards} shards, "
            f"metric={sh.stats()['metric']}"
        )
        return 0

    ids = _load_ids(args.ids) if args.ids else None
    idx = TQEIndex.create(
        emb,
        output_dim=args.output_dim,
        bits=args.bits,
        seed=args.seed,
        rotation=args.rotation,
        whiten=args.whiten,
        metric=args.metric,
        keep_originals=not args.no_originals,
        ids=ids,
    )
    idx.save(args.out)
    print(f"wrote {args.out}")
    print(_index_stats_summary(idx.stats()))
    return 0


def _open_index_for_search(path: str, mmap: bool):
    """Open a single index or a shard set for search, auto-detecting a manifest."""
    import os

    from .index import TQEIndex

    manifest = path
    if os.path.isdir(path):
        manifest = os.path.join(path, "manifest.json")
    if manifest.endswith("manifest.json") and os.path.exists(manifest):
        from .sharded_index import ShardedIndex

        return ShardedIndex.open(manifest, mmap=mmap)
    return TQEIndex.open(path, mmap=mmap)


def _cmd_index_add(args: argparse.Namespace) -> int:
    import numpy as np

    from .index import TQEIndex

    idx = TQEIndex.open(args.index)
    emb = np.asarray(np.load(args.embeddings))
    ids = _load_ids(args.ids) if args.ids else None
    new_ids = idx.add(emb, ids=ids)
    out = args.out or args.index
    idx.save(out)
    print(f"added {len(new_ids)} vectors -> {out}")
    print(_index_stats_summary(idx.stats()))
    return 0


def _cmd_index_delete(args: argparse.Namespace) -> int:
    from .index import TQEIndex

    idx = TQEIndex.open(args.index)
    deleted = idx.delete(_load_ids(args.ids))
    out = args.out or args.index
    idx.save(out)
    print(f"tombstoned {deleted} rows -> {out}")
    print(_index_stats_summary(idx.stats()))
    return 0


def _cmd_index_compact(args: argparse.Namespace) -> int:
    from .index import TQEIndex

    idx = TQEIndex.open(args.index)
    reclaimed = idx.compact()
    out = args.out or args.index
    idx.save(out)
    print(f"compacted: reclaimed {reclaimed} rows -> {out}")
    print(_index_stats_summary(idx.stats()))
    return 0


def _cmd_index_migrate(args: argparse.Namespace) -> int:
    from .index import TQEIndex

    idx = TQEIndex.open(args.index)
    before = idx.stats()["format_version"]
    idx.migrate(args.to_version)
    out = args.out or args.index
    idx.save(out)
    print(f"migrated format v{before} -> v{args.to_version} -> {out}")
    return 0


def _cmd_index_search(args: argparse.Namespace) -> int:
    import numpy as np

    idx = _open_index_for_search(args.index, mmap=getattr(args, "mmap", False))
    q = np.asarray(np.load(args.queries))
    ids, scores = idx.search(
        q, k=args.k, rerank=args.rerank, block=getattr(args, "block", None)
    )
    doc = {
        "index": args.index,
        "k": args.k,
        "rerank": args.rerank,
        "n_queries": int(len(ids)),
        "results": [
            {
                "query": i,
                "ids": [int(v) for v in row if v >= 0],
                "scores": [float(s) for s, v in zip(srow, row) if v >= 0],
            }
            for i, (row, srow) in enumerate(zip(ids, scores))
        ],
    }
    summary = f"searched {len(ids)} queries, top-{args.k} ids per query"
    return 0 if _emit_doc(doc, args.out, args.format, summary) else 2


def _cmd_index_certify(args: argparse.Namespace) -> int:
    from turboquant_pro import __version__

    from .index import TQEIndex

    idx = TQEIndex.open(args.index)
    try:
        cert = idx.certify(sample=args.sample, n_anchors=args.anchors, seed=args.seed)
    except ValueError as e:
        print(f"index certify: {e}", file=sys.stderr)
        return 2
    c = cert.as_dict()
    passed = not cert.vacuous
    if args.min_tau is not None:
        import numpy as np

        passed = bool(np.isfinite(cert.tau_floor) and cert.tau_floor >= args.min_tau)
    doc = {
        "schema": "turboquant-pro/index-certificate",
        "schema_version": 1,
        "tool_version": __version__,
        "index": args.index,
        "metric": idx.stats()["metric"],
        "n_live": idx.stats()["n_live"],
        "sample": args.sample,
        "certificate": c,
        "passed": passed,
    }
    summary = (
        f"index certificate: tau floor >= {c['tau_floor']}, "
        f"vacuous={cert.vacuous}, passed={passed}"
    )
    ok = _emit_doc(doc, args.out, args.format, summary)
    if not ok:
        return 2
    return 0 if passed else 1


def _cmd_index_drift(args: argparse.Namespace) -> int:
    import numpy as np

    from .index import TQEIndex

    idx = TQEIndex.open(args.index)
    emb = np.asarray(np.load(args.embeddings))
    report = idx.drift(emb, var_drop_threshold=args.threshold)
    doc = {"index": args.index, "drift": report.as_dict()}
    summary = (
        f"drift: retained var {report.retained_var_fit:.3f} (fit) -> "
        f"{report.retained_var_new:.3f} (new), drop {report.retained_var_drop:.3f}, "
        f"mean-shift {report.mean_shift:.3f}, stale={report.stale}"
    )
    ok = _emit_doc(doc, args.out, args.format, summary)
    if not ok:
        return 2
    return 1 if report.stale else 0


def _cmd_index_info(args: argparse.Namespace) -> int:
    from .index import TQEIndex, index_info

    info = index_info(args.index)
    info["stats"] = TQEIndex.open(args.index).stats()
    summary = _index_stats_summary(info["stats"])
    return 0 if _emit_doc(info, args.out, args.format, summary) else 2


def _cmd_anatomy(args: argparse.Namespace) -> int:
    import numpy as np

    from .anatomy import hub_anatomy

    base = np.load(args.npy, mmap_mode="r")
    base = np.asarray(base[: args.limit] if args.limit else base)
    queries = None
    if args.queries:
        queries = np.asarray(np.load(args.queries, mmap_mode="r"))
    doc = hub_anatomy(base, queries, k=args.k, hub_quantile=args.hub_quantile)
    c, a = doc["hub_vs_all_median_centrality"]
    summary = (
        f"{doc['battery']} k={doc['k']} n={doc['n_base']}: "
        f"count skew {doc['count_skew']:.2f} max {doc['count_max']:.0f}; "
        f"corr(count, -d_k) {doc['corr_count_neg_dk']:+.2f} "
        f"corr(count, centrality) {doc['corr_count_centrality']:+.2f}; "
        f"hub/all centrality {c:.3f}/{a:.3f} — "
        "density-driven hubs read high corr(-d_k) with hub medians near the "
        "population; centrality super-hubs read the opposite"
    )
    return 0 if _emit_doc(doc, args.out, args.format, summary) else 2


def _cmd_hubdiff(args: argparse.Namespace) -> int:
    import numpy as np

    from .anatomy import hub_differential, knn_exact

    if args.exact and args.approx:
        exact = np.asarray(np.load(args.exact, mmap_mode="r"))
        approx = np.asarray(np.load(args.approx, mmap_mode="r"))
        n_base = args.n_base or int(max(exact.max(), approx.max())) + 1
    elif args.original and args.reconstructed:
        orig = np.asarray(np.load(args.original, mmap_mode="r"), dtype=np.float32)
        recon = np.asarray(np.load(args.reconstructed, mmap_mode="r"), dtype=np.float32)
        if orig.shape != recon.shape:
            print(
                f"shape mismatch: original {orig.shape} vs "
                f"reconstructed {recon.shape}",
                file=sys.stderr,
            )
            return 2
        if args.queries:
            q = np.asarray(np.load(args.queries, mmap_mode="r"), dtype=np.float32)
            _, exact = knn_exact(orig, q, args.k)
            _, approx = knn_exact(recon, q, args.k)
        else:  # corpus->corpus: queries are the original rows in both searches
            _, exact = knn_exact(orig, orig, args.k, exclude_self=True)
            _, approx = knn_exact(recon, orig, args.k)
        n_base = len(orig)
    else:
        print(
            "hubdiff needs either --exact/--approx neighbour-id .npy arrays "
            "(system-agnostic) or --original/--reconstructed embeddings "
            "(tqp-compression differential)",
            file=sys.stderr,
        )
        return 2
    doc = hub_differential(
        exact,
        approx,
        n_base,
        k=args.k,
        hub_quantile=args.hub_quantile,
        anti_quantile=args.anti_quantile,
    )
    gap = doc["recall_at_k"] - doc["anti_hub_recall"]
    summary = (
        f"recall@{doc['k']} {doc['recall_at_k']:.4f} (p05 {doc['recall_p05']:.4f}) | "
        f"hub-rank corr {doc['hub_rank_corr']:+.3f} "
        f"hub-set Jaccard {doc['hub_set_jaccard']:.3f} | "
        f"anti-hub recall {doc['anti_hub_recall']:.4f} "
        f"({doc['anti_hub_query_frac'] * 100:.1f}% of queries)"
        + (
            f" — WARNING: anti-hub gap {gap:.3f}; the mean is hiding a tail"
            if np.isfinite(gap) and gap > args.gap_warn
            else ""
        )
    )
    ok = _emit_doc(doc, args.out, args.format, summary)
    if not ok:
        return 2
    if args.min_anti_recall is not None and not (
        doc["anti_hub_recall"] >= args.min_anti_recall
    ):
        return 1
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    import json
    import os

    from .query import (
        AnalyzeStmt,
        QuerySyntaxError,
        SelectStmt,
        analyze,
        catalog_path,
        execute,
        explain,
        parse,
    )

    try:
        stmt = parse(args.statement)
    except QuerySyntaxError as e:
        print(f"query: {e}", file=sys.stderr)
        return 2

    if isinstance(stmt, AnalyzeStmt):
        doc = analyze(stmt)
        g = doc["geometry"]
        calib = doc.get("calibration")
        summary = (
            f"catalog -> {catalog_path(stmt.index)}: n={g['n']} dim={g['dim']} "
            f"id={g['id_twonn'] if g['id_twonn'] is None else round(g['id_twonn'], 1)} "
            f"eff_rank={g['eff_rank']:.1f} bb_skew={g['hubness']['bb_skew']:.2f}; "
            + (
                f"calibrated {len(calib['sweep'])} operating points at k={calib['k']}"
                if calib
                else "NO calibration (no originals) — planner unavailable"
            )
        )
        return 0 if _emit_doc(doc, args.out, args.format, summary) else 2

    assert isinstance(stmt, SelectStmt)
    catalog = None
    cpath = catalog_path(stmt.index)
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            catalog = json.load(f)

    if stmt.explain:
        if catalog is None:
            print(
                f"query: no statistics catalog for {stmt.index!r}; run "
                f"tqp query \"ANALYZE INDEX '{stmt.index}'\" first",
                file=sys.stderr,
            )
            return 2
        doc = explain(stmt, catalog)
        plan = doc["plan"]
        pred = plan.get("predicted") or {}
        summary = (
            f"plan: rerank={plan['rerank']} ({plan['chosen_by']})"
            + (
                f" -> predicted recall {pred['recall']:.3f} "
                f"@ ~{pred['latency_us']:.0f}us/query"
                if pred
                else ""
            )
            + ("  [TARGET UNREACHABLE]" if plan.get("target_unreachable") else "")
        )
        return 0 if _emit_doc(doc, args.out, args.format, summary) else 2

    if not args.queries:
        print("query: SELECT needs --queries q.npy to bind :q", file=sys.stderr)
        return 2
    import numpy as np

    q = np.asarray(np.load(args.queries))
    try:
        doc = execute(stmt, q, catalog)
    except RuntimeError as e:
        print(f"query: {e}", file=sys.stderr)
        return 2
    summary = (
        f"{doc['n_queries']} queries, top-{stmt.limit}, rerank={doc['plan']['rerank']} "
        f"({doc['plan']['chosen_by']}), ~{doc['mean_latency_us']:.0f}us/query"
    )
    return 0 if _emit_doc(doc, args.out, args.format, summary) else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tqp",
        description="turboquant-pro unified CLI: trace, probe, plan, certify, "
        "replay, monitor, plugins.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print the installed version").set_defaults(
        func=_cmd_version
    )

    # plugin (nested)
    pl = sub.add_parser("plugin", help="quantizer plugin registry + conformance")
    plsub = pl.add_subparsers(dest="plugin_command", required=True)
    pl_list = plsub.add_parser("list", help="list registered plugins")
    pl_list.add_argument(
        "--target", help="filter by target (weight/kv_key/kv_value/embedding)"
    )
    pl_list.add_argument(
        "-v", "--verbose", action="store_true", help="show descriptions"
    )
    pl_list.set_defaults(func=_cmd_plugin_list)
    pl_conf = plsub.add_parser("conformance", help="run the conformance kit")
    pl_conf.add_argument(
        "names", nargs="*", help="plugin names (default: all registered)"
    )
    pl_conf.add_argument("--target", help="only check plugins for this target")
    pl_conf.add_argument("--heads", type=int, default=4, help="KV heads H (default 4)")
    pl_conf.add_argument(
        "--seq", type=int, default=96, help="sequence length S (default 96)"
    )
    pl_conf.add_argument("--dim", type=int, default=64, help="head dim D (default 64)")
    pl_conf.add_argument(
        "--shape", help="explicit comma-separated sample shape (overrides H/S/D)"
    )
    pl_conf.set_defaults(func=_cmd_plugin_conformance)

    # trace
    tr = sub.add_parser(
        "trace", help="operator-regime -> (A2) discipline for an HF model"
    )
    tr.add_argument("model", help="Hugging Face model id or local path")
    tr.add_argument(
        "--target",
        default="weight",
        help="quantization target: weight | kv_activation (default weight)",
    )
    tr.add_argument(
        "--prefer",
        default="auto",
        choices=["auto", "structural", "fx"],
        help="tracing strategy (default auto)",
    )
    tr.add_argument(
        "--trust-remote-code", action="store_true", help="allow custom model code"
    )
    tr.add_argument("-v", "--verbose", action="store_true", help="per-tensor table")
    tr.set_defaults(func=_cmd_trace)

    # probe — (A2) consumer-metric quantizer-family selection
    pr = sub.add_parser(
        "probe", help="(A2) consumer-metric quantizer-family probe (a2_probe)"
    )
    pr.add_argument("--npy", help="path to a .npy batch, shape (n, d) or (..., d)")
    pr.add_argument(
        "--demo",
        choices=["isotropic", "dc_offset"],
        help="use a labeled SYNTHETIC batch instead of --npy (illustration)",
    )
    pr.add_argument(
        "--consumer",
        choices=_PROBE_CONSUMERS,
        default="cosine",
        help="downstream metric (default cosine; attention keys use "
        "attention_logits)",
    )
    pr.add_argument("--queries", help="optional .npy of query vectors (n, d)")
    pr.add_argument("--bits", type=int, default=4, help="probe bit budget (default 4)")
    pr.add_argument("--seed", type=int, default=0, help="determinism seed (default 0)")
    pr.add_argument("--json", action="store_true", help="emit the result as JSON")
    pr.set_defaults(func=_cmd_probe)

    # monitor — QualityMonitor metrics from original/reconstructed pairs
    mo = sub.add_parser(
        "monitor", help="QualityMonitor metrics (JSON/Prometheus) from orig/recon"
    )
    mo.add_argument("--original", required=True, help=".npy of original vectors")
    mo.add_argument(
        "--reconstructed", required=True, help=".npy of reconstructed vectors"
    )
    mo.add_argument(
        "--floor", type=float, default=0.95, help="quality floor (default 0.95)"
    )
    mo.add_argument(
        "--window", type=int, default=1000, help="rolling window (default 1000)"
    )
    mo.add_argument(
        "--tangential-floor",
        type=float,
        default=0.0,
        help="(A2) noncollapse floor: health also requires the median tangential "
        "fraction >= this (default 0.0 = off; the directional (A2) drift guard is "
        "always on)",
    )
    mo.add_argument(
        "--format",
        choices=["json", "prometheus", "text"],
        default="json",
        help="output format (default json)",
    )
    mo.set_defaults(func=_cmd_monitor)

    # certify — distribution-free rank certificate as certificate.json
    ce = sub.add_parser(
        "certify", help="emit a distribution-free rank certificate (certificate.json)"
    )
    ce.add_argument("--original", required=True, help=".npy of original embeddings")
    ce.add_argument(
        "--reconstructed", required=True, help=".npy of reconstructed embeddings"
    )
    ce.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="ranking metric to certify (default cosine)",
    )
    ce.add_argument(
        "--anchors", type=int, default=200, help="anchor rows to sample (default 200)"
    )
    ce.add_argument("--seed", type=int, default=0, help="anchor-sampling seed")
    ce.add_argument(
        "--min-tau",
        type=float,
        default=None,
        help="gate: exit 1 unless the Kendall tau floor is >= this "
        "(default: exit 1 only when the certificate is vacuous)",
    )
    ce.add_argument(
        "--out", help="write certificate.json here (default: stdout per --format)"
    )
    ce.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="stdout format when --out is not given (default json)",
    )
    ce.add_argument(
        "--task",
        help="declare the downstream target (e.g. 'recall@10 >= 0.995') — adds a "
        "task section to the certificate",
    )
    ce.add_argument(
        "--task-kind", default="retrieval", help="task kind (default retrieval)"
    )
    ce.add_argument(
        "--environment",
        action="store_true",
        help="stamp an environment section (tool/python/numpy/platform/git/hardware)",
    )
    ce.add_argument(
        "--limitation",
        action="append",
        help="record a scope caveat (repeatable) as a limitations entry",
    )
    ce.add_argument("--html", help="also write a readable HTML report here")
    ce.set_defaults(func=_cmd_certify)

    vf = sub.add_parser(
        "verify",
        help="verify a certificate.json — schema/self-consistency always, plus "
        "recompute vs inputs when --original/--reconstructed are given",
    )
    vf.add_argument(
        "certificate", help="path to a certificate.json (from `tqp certify`)"
    )
    vf.add_argument(
        "--original", help=".npy of original embeddings (enables recompute)"
    )
    vf.add_argument(
        "--reconstructed", help=".npy of reconstructed embeddings (enables recompute)"
    )
    vf.add_argument(
        "--atol", type=float, default=1e-6, help="abs tolerance on recomputed floors"
    )
    vf.add_argument(
        "--rtol", type=float, default=1e-4, help="rel tolerance on recomputed floors"
    )
    vf.add_argument("--out", help="write the verification report here")
    vf.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="stdout format when --out is not given (default text)",
    )
    vf.set_defaults(func=_cmd_verify)

    # plan (nested) — task-aware recipe planner
    pn = sub.add_parser("plan", help="task-aware recipe planner (embeddings | kv)")
    pnsub = pn.add_subparsers(dest="plan_command", required=True)
    pe = pnsub.add_parser(
        "embeddings", help="auto_compress + rank-certificate preview -> plan.json"
    )
    pe.add_argument("--embeddings", required=True, help=".npy of embeddings (n, d)")
    pe.add_argument(
        "--target",
        default="recall@10 >= 0.90",
        help="auto_compress target — prefer a measured recall@k "
        "(e.g. 'recall@10 >= 0.90'); 'cosine > 0.97' / 'ratio > 20' are "
        "reconstruction-only. The plan's overall acceptance is the "
        "rank-certificate preview, not this.",
    )
    pe.add_argument(
        "--max-bytes-per-vector",
        type=float,
        default=None,
        help="byte-budget constraint on the recommended recipe",
    )
    pe.add_argument(
        "--sample", type=int, default=100, help="quality-eval subsample (default 100)"
    )
    pe.add_argument(
        "--seed", type=int, default=42, help="determinism seed (default 42)"
    )
    pe.add_argument("--out", help="write plan.json here")
    pe.add_argument(
        "--format", choices=["json", "text"], default="json", help="stdout format"
    )
    pe.set_defaults(func=_cmd_plan_embeddings)
    pk = pnsub.add_parser("kv", help="AutoConfig KV key/value policy -> kv_plan.json")
    pk.add_argument("--model", required=True, help="model name (registry) or HF path")
    pk.add_argument(
        "--target",
        choices=["quality", "balanced", "compression", "extreme"],
        default="balanced",
        help="compression preset (default balanced)",
    )
    pk.add_argument(
        "--context",
        type=int,
        default=None,
        help="context-length override (max_seq_len)",
    )
    pk.add_argument("--out", help="write kv_plan.json here")
    pk.add_argument(
        "--format", choices=["json", "text"], default="json", help="stdout format"
    )
    pk.set_defaults(func=_cmd_plan_kv)

    # replay — execute claim reproductions from claims.yaml
    rp = sub.add_parser("replay", help="execute claim reproductions from claims.yaml")
    rp.add_argument("target", nargs="?", default=None, help="claim id, or 'all'")
    rp.add_argument(
        "--claims", default="claims.yaml", help="claims file (default claims.yaml)"
    )
    rp.add_argument("--track", help="filter 'all' by track (e.g. embedding | kv)")
    rp.add_argument(
        "--full", action="store_true", help="use full_command instead of command"
    )
    rp.add_argument("--list", action="store_true", help="list claims and exit")
    rp.add_argument(
        "--dry-run", action="store_true", help="show what would run, do not execute"
    )
    rp.add_argument("--cwd", default=".", help="working dir for commands (default .)")
    rp.add_argument(
        "--timeout", type=int, default=1800, help="per-command timeout s (default 1800)"
    )
    rp.add_argument("--json", action="store_true", help="emit the report as JSON")
    rp.add_argument("--out", help="write report.json here")
    rp.set_defaults(func=_cmd_replay)

    # index (nested) — production vector-index lifecycle
    ix = sub.add_parser("index", help="persisted vector-index lifecycle (TQE)")
    ixsub = ix.add_subparsers(dest="index_command", required=True)

    ic = ixsub.add_parser("create", help="fit + build an index from embeddings")
    ic.add_argument("--embeddings", required=True, help=".npy of embeddings (n, d)")
    ic.add_argument("--out", required=True, help="index path to write (e.g. index.tqe)")
    ic.add_argument(
        "--output-dim", type=int, default=None, help="PCA dim (default: full dim)"
    )
    ic.add_argument("--bits", type=int, default=3, help="quantizer bits (default 3)")
    ic.add_argument("--seed", type=int, default=42, help="determinism seed")
    ic.add_argument("--rotation", default="qr", choices=["qr", "hadamard"])
    ic.add_argument("--whiten", action="store_true", help="whiten PCA (hurts recall)")
    ic.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    ic.add_argument(
        "--no-originals",
        action="store_true",
        help="do not store fp32 originals (disables exact rerank + certify)",
    )
    ic.add_argument("--ids", help="ids file (one per line) or comma-separated list")
    ic.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="rows per shard (>0 writes a sharded index: --out is a directory of "
        "shard_*.tqe + manifest.json, one shared PCA basis; 0 = single file)",
    )
    ic.set_defaults(func=_cmd_index_create)

    ia = ixsub.add_parser("add", help="append new vectors (same basis, no refit)")
    ia.add_argument("index", help="index path")
    ia.add_argument("--embeddings", required=True, help=".npy of new embeddings")
    ia.add_argument("--ids", help="ids for the new vectors (file or comma list)")
    ia.add_argument("--out", help="write here instead of in place")
    ia.set_defaults(func=_cmd_index_add)

    idel = ixsub.add_parser("delete", help="tombstone rows by external id")
    idel.add_argument("index", help="index path")
    idel.add_argument("--ids", required=True, help="ids file or comma-separated list")
    idel.add_argument("--out", help="write here instead of in place")
    idel.set_defaults(func=_cmd_index_delete)

    icomp = ixsub.add_parser("compact", help="physically drop tombstoned rows")
    icomp.add_argument("index", help="index path")
    icomp.add_argument("--out", help="write here instead of in place")
    icomp.set_defaults(func=_cmd_index_compact)

    imig = ixsub.add_parser("migrate", help="upgrade the on-disk format version")
    imig.add_argument("index", help="index path")
    imig.add_argument(
        "--to-version", type=int, required=True, help="target format version (e.g. 2)"
    )
    imig.add_argument("--out", help="write here instead of in place")
    imig.set_defaults(func=_cmd_index_migrate)

    isea = ixsub.add_parser("search", help="top-k search (excludes tombstones)")
    isea.add_argument("index", help="index path")
    isea.add_argument("--queries", required=True, help=".npy of query vectors (n, d)")
    isea.add_argument("--k", type=int, default=10, help="neighbours per query")
    isea.add_argument(
        "--rerank",
        type=int,
        default=0,
        help="exact-rerank oversample factor (needs stored originals for exact)",
    )
    isea.add_argument(
        "--mmap",
        action="store_true",
        help="memory-map the index (bounded RAM; for indexes too large to load). "
        "A directory or manifest.json path is searched as a shard set.",
    )
    isea.add_argument(
        "--block",
        type=int,
        default=None,
        help="row-block size for the bounded-memory blocked search",
    )
    isea.add_argument("--out", help="write results.json here")
    isea.add_argument("--format", choices=["json", "text"], default="json")
    isea.set_defaults(func=_cmd_index_search)

    icert = ixsub.add_parser(
        "certify", help="rank certificate over stored originals (keep-originals on)"
    )
    icert.add_argument("index", help="index path")
    icert.add_argument("--sample", type=int, default=512, help="live rows to sample")
    icert.add_argument("--anchors", type=int, default=200, help="certificate anchors")
    icert.add_argument("--seed", type=int, default=0, help="sampling seed")
    icert.add_argument(
        "--min-tau",
        type=float,
        default=None,
        help="gate: exit 1 unless the Kendall tau floor >= this",
    )
    icert.add_argument("--out", help="write certificate.json here")
    icert.add_argument("--format", choices=["json", "text"], default="json")
    icert.set_defaults(func=_cmd_index_certify)

    idr = ixsub.add_parser("drift", help="check whether the PCA basis is stale")
    idr.add_argument("index", help="index path")
    idr.add_argument(
        "--embeddings", required=True, help=".npy of new-distribution vectors"
    )
    idr.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="stale if retained-variance drop exceeds this (default 0.05)",
    )
    idr.add_argument("--out", help="write drift.json here")
    idr.add_argument("--format", choices=["json", "text"], default="json")
    idr.set_defaults(func=_cmd_index_drift)

    iinf = ixsub.add_parser("info", help="container + stats summary")
    iinf.add_argument("index", help="index path")
    iinf.add_argument("--out", help="write info.json here")
    iinf.add_argument("--format", choices=["json", "text"], default="json")
    iinf.set_defaults(func=_cmd_index_info)

    qy = sub.add_parser(
        "query",
        help="SQL-ish workload interface: ANALYZE INDEX / EXPLAIN SELECT / SELECT",
        description=(
            "One statement per invocation. ANALYZE INDEX 'x.tqe' [USING QUERIES "
            "'q.npy'] builds the statistics catalog (geometry + a measured "
            "recall/latency calibration sweep). EXPLAIN SELECT ... WITH (RECALL "
            ">= 0.95) shows the calibration-based plan. SELECT id, score FROM "
            "'x.tqe' ORDER BY COSINE(:q) LIMIT 10 WITH (RECALL >= 0.95[, "
            "CERTIFY]) executes it (:q binds --queries)."
        ),
    )
    qy.add_argument("statement", help="the SQL-ish statement (quote it)")
    qy.add_argument("--queries", help=".npy query vectors bound to :q (SELECT)")
    qy.add_argument("--out", help="write the result document to this JSON path")
    qy.add_argument(
        "--format",
        choices=("json", "summary"),
        default="summary",
        help="stdout format when --out is not given (default: summary)",
    )
    qy.set_defaults(func=_cmd_query)

    # anatomy — the hub anatomy vector (scalar hubness is non-identifying)
    an = sub.add_parser(
        "anatomy",
        help="hub anatomy vector of a corpus (what the hubs ARE, not just skew)",
        description=(
            "Reverse-kNN count tail plus what the hubs are: rank correlations "
            "of the count with centrality, local density (-d_k) and "
            "nearest-pair distance (-d_1), and hub-vs-population medians. Two "
            "corpora can share the same scalar hubness with opposite anatomy "
            "and opposite ANN behaviour — report the anatomy, not the scalar."
        ),
    )
    an.add_argument("--npy", required=True, help=".npy corpus embeddings")
    an.add_argument(
        "--queries",
        help=".npy query embeddings (query->corpus battery; default corpus->corpus)",
    )
    an.add_argument("--k", type=int, default=10, help="neighbourhood size")
    an.add_argument(
        "--limit", type=int, help="use only the first N corpus rows (memory guard)"
    )
    an.add_argument(
        "--hub-quantile",
        type=float,
        default=0.99,
        help="count quantile defining the hub set (default 0.99)",
    )
    an.add_argument("--out", help="write anatomy.json here")
    an.add_argument(
        "--format",
        choices=["json", "summary"],
        default="summary",
        help="stdout format when --out is not given (default summary)",
    )
    an.set_defaults(func=_cmd_anatomy)

    # hubdiff — compression differential oracle: the tail the mean recall hides
    hd = sub.add_parser(
        "hubdiff",
        help="differential oracle: exact vs compressed top-k — hub-rank "
        "divergence + anti-hub recall (the tail aggregate recall hides)",
        description=(
            "Either --original/--reconstructed embeddings (tqp-compression "
            "differential; queries default to corpus->corpus) or "
            "--exact/--approx neighbour-id .npy arrays from ANY two systems "
            "(HNSW vs exact, two build orders, two shardings). Acceptance "
            "stays coherent with the rest of tqp: rank fidelity at the TAIL, "
            "never an aggregate alone — gate with --min-anti-recall."
        ),
    )
    hd.add_argument("--original", help=".npy of original embeddings")
    hd.add_argument("--reconstructed", help=".npy of reconstructed embeddings")
    hd.add_argument("--queries", help=".npy query embeddings (with --original)")
    hd.add_argument("--exact", help=".npy (n_q, >=k) exact neighbour ids")
    hd.add_argument("--approx", help=".npy (n_q, >=k) approximate neighbour ids")
    hd.add_argument("--n-base", type=int, help="base row count (with --exact/--approx)")
    hd.add_argument("--k", type=int, default=10, help="neighbourhood size")
    hd.add_argument("--hub-quantile", type=float, default=0.99)
    hd.add_argument(
        "--anti-quantile",
        type=float,
        default=0.10,
        help="exact-count quantile defining anti-hub rows (default 0.10)",
    )
    hd.add_argument(
        "--gap-warn",
        type=float,
        default=0.05,
        help="warn when recall@k - anti_hub_recall exceeds this (default 0.05)",
    )
    hd.add_argument(
        "--min-anti-recall",
        type=float,
        default=None,
        help="gate: exit 1 unless anti-hub recall >= this",
    )
    hd.add_argument("--out", help="write hubdiff.json here")
    hd.add_argument(
        "--format",
        choices=["json", "summary"],
        default="summary",
        help="stdout format when --out is not given (default summary)",
    )
    hd.set_defaults(func=_cmd_hubdiff)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
