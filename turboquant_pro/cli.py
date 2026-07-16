"""``tqp`` — the unified turboquant-pro CLI (Phase 1 of the next-level roadmap).

Surfaces capability that already exists in the library behind one command:

    tqp version
    tqp plugin list                 # registered quantizer plugins
    tqp plugin conformance [names]  # run the container-contract conformance kit
    tqp trace <hf-model>            # operator regime -> (A2) discipline per tensor
    tqp probe --npy keys.npy        # (A2) consumer-metric quantizer-family probe
    tqp monitor --original o.npy --reconstructed r.npy   # QualityMonitor metrics

Subcommands that are still roadmap items (``plan``, ``certify``, ``replay``)
print what they will do and exit 2 — the surface is visible without
overclaiming maturity. See ``docs/turboquant_pro_next_level_roadmap.md``.

Registered as the ``tqp`` console script; ``turboquant-pro`` remains the
AutoConfig entry point.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

ROADMAP = "docs/turboquant_pro_next_level_roadmap.md"


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

    mon = QualityMonitor(quality_floor=args.floor, window_size=args.window)
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
    # Exit non-zero when quality is below the floor, so `tqp monitor` is a gate.
    return 0 if stats["is_healthy"] else 1


def _make_stub(name: str, phase: str, what: str):
    def run(args: argparse.Namespace) -> int:
        print(f"`tqp {name}` is not implemented yet.")
        print(f"Planned: {what} ({phase}).")
        print(f"See {ROADMAP}.")
        return 2

    return run


# ------------------------------------------------------------------ parser
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
        "--format",
        choices=["json", "prometheus", "text"],
        default="json",
        help="output format (default json)",
    )
    mo.set_defaults(func=_cmd_monitor)

    # honest stubs for the not-yet-built surface
    stubs = {
        "plan": ("Phase 4", "task-aware recipe compiler (auto_compress / AutoConfig)"),
        "certify": ("Phase 2", "emit a machine-readable certificate.json"),
        "replay": ("Phase 3", "executable claim replay from claims.yaml"),
    }
    for name, (phase, what) in stubs.items():
        sp = sub.add_parser(name, help=f"[{phase}] {what}")
        sp.add_argument("args", nargs="*", help=argparse.SUPPRESS)
        sp.set_defaults(func=_make_stub(name, phase, what))

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
