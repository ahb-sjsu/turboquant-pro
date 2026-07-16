"""``tqp`` — the unified turboquant-pro CLI (Phase 1 of the next-level roadmap).

Surfaces capability that already exists in the library behind one command:

    tqp version
    tqp plugin list                 # registered quantizer plugins
    tqp plugin conformance [names]  # run the container-contract conformance kit
    tqp trace <hf-model>            # operator regime -> (A2) discipline per tensor

Subcommands that are still roadmap items (``plan``, ``certify``, ``replay``,
``monitor``, ``probe``) print what they will do and exit 2 — the surface is
visible without overclaiming maturity. See
``docs/turboquant_pro_next_level_roadmap.md``.

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

    # honest stubs for the not-yet-built surface
    stubs = {
        "plan": ("Phase 4", "task-aware recipe compiler (auto_compress / AutoConfig)"),
        "certify": ("Phase 2", "emit a machine-readable certificate.json"),
        "replay": ("Phase 3", "executable claim replay from claims.yaml"),
        "monitor": ("Phase 1", "serve QualityMonitor metrics (JSON/Prometheus)"),
        "probe": ("Phase 1", "a2_probe consumer-metric quantizer selection"),
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
