"""Build the anonymized supplement for the submission.

    python paper/tmlr_operator/scripts/build_supplement.py OUT_DIR [--zip FILE]

Copies the experiment code, the run records, the reproduction notebook, the
number-derivation scripts and the Lean file into OUT_DIR, in the repository's
own layout so that the notebook finds its records. Source files lose their
licence header and every mention of the library; the library modules the
scripts import ship as the package `opquant`. Run records are copied byte for
byte, because the derivation scripts print a hash of each one.

The build then scans every file it wrote for identifying strings and fails if
it finds one. It needs only the standard library.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LIB = "turboquant" + "_pro"
PKG = "opquant"

EXPERIMENTS = [
    "agreement_reference_repeats.py",
    "behavioral_metric_demo.py",
    "exp_io.py",
    "k_quantizer_control.py",
    "k_row_protection_controls.py",
    "k_wavelength_probe.py",
    "llamacpp_factorial.py",
    "matched_bit_projection_sensitivity.py",
    "router_error_margin.py",
    "routing_margin_logits.py",
    "score_error_decomposition.py",
    "tracer_accuracy.py",
    "validate_rope_aware_k.py",
]
BENCHMARKS = [
    "mamba_decay_grid_census.py",
    "mamba_decay_tau_grid.py",
    "rwkv_decay_grid.py",
    "validate_mamba_decay.py",
    "validate_mixtral_routing.py",
    "validate_olmoe_routing.py",
]
LIB_MODULES = [
    "behavioral_agreement.py",
    "model_compress.py",
    "operator_sensitivity.py",
    "operator_trace.py",
]
BENCH_RECORDS = re.compile(
    r"^(results_(mamba_decay|rwkv_decay|mixtral_routing|olmoe_routing).*\.json"
    r"|router_error_margin_.*\.json|routing_margin_logits_.*\.json"
    r"|routing_margins_.*\.npz)$"
)
RM_SKIP = {"nrp_usage_observed.json"}
PAPER_FILES = [
    "reproduce.ipynb",
    "lean/OperatorChecks.lean",
    "lean/BUILD-RECORD.txt",
    "scripts/build_notebook.py",
    "scripts/derive_numbers.py",
    "scripts/derive_ratios.py",
    "scripts/derive_e_numbers.py",
    "scripts/make_fig_strata.py",
    "scripts/numbers_derived.txt",
    "scripts/numbers_derived_e.txt",
]

HEADER = re.compile(
    r"\A# [^\n]*\n# Copyright[^\n]*\n# MIT License\n\n?", re.MULTILINE
)
# Applied in order to every source file (not to run records).
REWRITES = [
    (re.compile(LIB), PKG),
    (re.compile(r"TQP_REPO"), "SUPPLEMENT_ROOT"),
    (re.compile(r"TurboQuant Pro|TurboQuant-Pro|TurboQuant", re.I), "the library"),
    (re.compile(r"NRP Nautilus batch Job"), "cluster batch job"),
    (re.compile(r"nrp_tmlr"), "staged"),
    (re.compile(r"\ban NRP\b"), "a cluster"),
    (re.compile(r"\bNRP\b"), "the cluster"),
    (re.compile(r"Atlas GPU 1"), "the workstation GPU"),
    (re.compile(r"Atlas's"), "the workstation's"),
    (re.compile(r"\bthe Atlas\b"), "the workstation"),
    (re.compile(r"\bAtlas GV100\b"), "workstation GV100"),
    (re.compile(r"\bAtlas\b"), "the workstation"),
    (re.compile(r"/home/claude/llama\.cpp"), "./llama.cpp"),
    (re.compile(r"\(Bond, 2026\)"), "(2026)"),
]
# One record names the cluster in a hardware field. The supplement's copy has
# that field rewritten, and the saved derivation output carries the new hash.
RECORD_FIX = (
    "validate_rope_aware_k_Llama-3.2-3B.json",
    b"(NRP Nautilus batch Job)",
    b"(cluster batch job)",
)
# The helpers that stage weights and tokens on a cluster pod are not shipped:
# they depend on a package that would identify the authors. Only the optional
# NB_STRIPES path of exp_io.py uses them.
STAGED: list[str] = []
BANNED = re.compile(
    r"turboquant|tqp|\bbond\b|andrew|ahb|sjsu|sonoma|abptl|agi[.-]hpc|atlas"
    r"|erebus|nautilus|\bnrp\b|ssu-|duckdns|tailscale|/home/|/archive|(?-i:\b[A-Z]:\\+(Users|source|tools|Program))"
    r"|100\.68\.|192\.168\.|zenodo|hf_[A-Za-z0-9]{20}|github\.com/(?!ggml-org|state-spaces"
    r"|Dao-AILab|alxndrTL|huggingface|leanprover)|claude|anthropic|@[a-z0-9-]+\.(com|edu|org)",
    re.I,
)
TEXT_EXT = {".py", ".txt", ".md", ".json", ".ipynb", ".lean"}


def scrub(text: str) -> str:
    text = HEADER.sub("", text)
    for pat, new in REWRITES:
        text = pat.sub(new, text)
    return text


def put_source(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as fh:
        text = fh.read().decode("utf-8").replace("\r\n", "\n")
    with open(dst, "wb") as fh:
        fh.write(scrub(text).encode("utf-8"))


def put_raw(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> int:
    out = os.path.abspath(sys.argv[1])
    zip_path = sys.argv[sys.argv.index("--zip") + 1] if "--zip" in sys.argv else None
    assert not os.path.exists(out) or not os.listdir(out), f"{out} is not empty"

    for n in EXPERIMENTS:
        put_source(os.path.join(ROOT, "experiments", n), os.path.join(out, "experiments", n))
    for n in STAGED:
        put_source(
            os.path.join(ROOT, "experiments", "nrp_tmlr", n),
            os.path.join(out, "experiments", "staged", n),
        )
    for n in BENCHMARKS:
        put_source(os.path.join(ROOT, "benchmarks", n), os.path.join(out, "benchmarks", n))
    for n in LIB_MODULES:
        put_source(os.path.join(ROOT, LIB, n), os.path.join(out, PKG, n))
    with open(os.path.join(out, PKG, "__init__.py"), "w", newline="\n") as fh:
        fh.write('"""Library modules the experiment scripts import."""\n')

    rm = os.path.join(ROOT, "experiments", "results_matched_bit")
    for base, _dirs, files in os.walk(rm):
        for n in files:
            if n in RM_SKIP:
                continue
            src = os.path.join(base, n)
            rel = os.path.relpath(src, ROOT)
            (put_source if n.endswith(".md") else put_raw)(src, os.path.join(out, rel))
    for n in sorted(os.listdir(os.path.join(ROOT, "benchmarks"))):
        if BENCH_RECORDS.match(n):
            put_raw(os.path.join(ROOT, "benchmarks", n), os.path.join(out, "benchmarks", n))

    for n in PAPER_FILES:
        src = os.path.join(ROOT, "paper", "tmlr_operator", n)
        dst = os.path.join(out, "paper", "tmlr_operator", n)
        if n.endswith((".py", ".ipynb")):
            put_source(src, dst)
        else:
            put_raw(src, dst)
    put_source(os.path.join(HERE, "SUPPLEMENT-README.md"), os.path.join(out, "README.md"))

    name, old, new = RECORD_FIX
    rp = os.path.join(out, "experiments", "results_matched_bit", name)
    with open(rp, "rb") as fh:
        raw = fh.read()
    assert raw.count(old) == 1
    h_old = hashlib.sha256(raw).hexdigest()[:16].encode()
    raw = raw.replace(old, new)
    with open(rp, "wb") as fh:
        fh.write(raw)
    h_new = hashlib.sha256(raw).hexdigest()[:16].encode()
    dp = os.path.join(out, "paper", "tmlr_operator", "scripts", "numbers_derived.txt")
    with open(dp, "rb") as fh:
        d = fh.read()
    assert d.count(old) == 1 and d.count(h_old) == 1, (d.count(old), h_old)
    with open(dp, "wb") as fh:
        fh.write(d.replace(old, new).replace(h_old, h_new))

    hits, count = [], 0
    for base, _dirs, files in os.walk(out):
        for n in files:
            count += 1
            path = os.path.join(base, n)
            rel = os.path.relpath(path, out).replace(os.sep, "/")
            if BANNED.search(rel):
                hits.append((rel, 0, "file name"))
            if os.path.splitext(n)[1] not in TEXT_EXT:
                continue
            with open(path, "rb") as fh:
                text = fh.read().decode("utf-8", errors="replace")
            for i, line in enumerate(text.split("\n"), 1):
                for m in BANNED.finditer(line):
                    lo = max(0, m.start() - 40)
                    hits.append((rel, i, line[lo : m.end() + 40].strip()))
    for rel, i, ctx in hits[:200]:
        print(f"IDENTIFYING {rel}:{i}: {ctx}")
    print(f"{count} files written, {len(hits)} identifying strings")
    if hits:
        return 1
    if zip_path:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for base, _dirs, files in sorted(os.walk(out)):
                for n in sorted(files):
                    path = os.path.join(base, n)
                    arc = os.path.relpath(path, out).replace(os.sep, "/")
                    info = zipfile.ZipInfo("supplement/" + arc, (2026, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    with open(path, "rb") as fh:
                        z.writestr(info, fh.read())
        print(f"zip written: {os.path.getsize(zip_path) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
