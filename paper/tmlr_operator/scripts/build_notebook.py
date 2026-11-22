"""Write paper/tmlr_operator/reproduce.ipynb. Run from the repository root:

    python paper/tmlr_operator/scripts/build_notebook.py

The notebook rebuilds every number the paper reports from the committed JSON records and
compares it with the value printed in the paper. It does not rerun the models.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Reproduction notebook: Quantization Sensitivity Follows the Consuming Operator

This notebook rebuilds every number reported in the paper from the JSON record of each run and
compares it with the value printed in the paper. A check fails if the two differ by more than
the rounding of the printed value.

**What it does not do.** It does not rerun the models. The records were produced by the scripts
listed in the last section, which need a GPU. It also does not rebuild the Lean file; the
section on formal checks prints the build record and the command.

Run it from anywhere inside the repository. It needs only the Python standard library.
""")

code(r'''
import glob, json, math, os, re, shutil, statistics, subprocess, sys, tempfile

root = os.path.abspath(".")
while not os.path.isdir(os.path.join(root, "experiments", "results_matched_bit")):
    parent = os.path.dirname(root)
    assert parent != root, "run this notebook inside the repository"
    root = parent
RM = os.path.join(root, "experiments", "results_matched_bit")
BM = os.path.join(root, "benchmarks")
PAPER = os.path.join(root, "paper", "tmlr_operator")

def rec(folder, name):
    with open(os.path.join(folder, name), "rb") as fh:
        return json.loads(fh.read())

CHECKS = []

def check(where, printed, got, tol):
    """Record one comparison: the paper prints `printed`, the record gives `got`."""
    ok = abs(float(got) - float(printed)) <= tol * (1 + 1e-9)
    CHECKS.append((where, printed, float(got), ok))
    return ok

def report(prefix):
    rows = [c for c in CHECKS if c[0].startswith(prefix)]
    bad = [c for c in rows if not c[3]]
    print(f"{prefix}: {len(rows) - len(bad)} of {len(rows)} checks pass")
    for where, printed, got, ok in bad:
        print("  FAIL", where, "paper", printed, "record", got)

print("repository root found; records:", len(glob.glob(os.path.join(RM, "*.json"))), "in results_matched_bit")
''')

md("""
## Table 1: decision-level agreement, with intervals from resampling whole sequences

Record: `experiments/results_matched_bit/agreement_repeats_<model>.json`. The intervals are the
record's own bootstrap over sequences (2,000 resamples).
""")

code(r'''
T1 = {  # model: bits: (accuracy, dAcc, [lo, hi], r, c, 1-A, [lo, hi], D-D0, [lo, hi])
 "Qwen2.5-1.5B-Instruct": {"base": 0.462,
   "8": (0.462, 0.000, (-0.002, 0.001), 0.003, 0.004, 0.022, (0.018, 0.026), 0.008, (0.004, 0.012)),
   "4": (0.367, 0.095, (0.085, 0.106), 0.131, 0.036, 0.409, (0.388, 0.429), 0.395, (0.376, 0.413)),
   "3": (0.002, 0.459, (0.439, 0.481), 0.460, 0.001, 0.996, (0.995, 0.998), 0.982, (0.979, 0.985))},
 "Mistral-7B-v0.1": {"base": 0.537,
   "8": (0.538, 0.000, (-0.001, 0.000), 0.002, 0.002, 0.012, (0.010, 0.013), 0.005, (0.003, 0.007)),
   "4": (0.495, 0.043, (0.037, 0.048), 0.072, 0.029, 0.232, (0.224, 0.240), 0.226, (0.217, 0.234)),
   "3": (0.000, 0.537, (0.519, 0.556), 0.537, 0.000, 0.996, (0.996, 0.996), 0.990, (0.988, 0.991))},
 "Llama-3.2-3B": {"base": 0.496,
   "8": (0.496, 0.000, (-0.002, 0.002), 0.003, 0.003, 0.022, (0.019, 0.025), 0.010, (0.008, 0.013)),
   "4": (0.429, 0.067, (0.057, 0.078), 0.100, 0.033, 0.319, (0.305, 0.332), 0.307, (0.293, 0.320)),
   "3": (0.054, 0.442, (0.420, 0.463), 0.451, 0.009, 0.919, (0.910, 0.927), 0.907, (0.899, 0.915))},
}
SEQS = {"Qwen2.5-1.5B-Instruct": (24, 6120), "Mistral-7B-v0.1": (48, 12240), "Llama-3.2-3B": (32, 8160)}
D0 = {  # ranges over three seed pairs, in percent, at dither 0.02, 0.05, 0.1 of a step
 "Llama-3.2-3B": [(1.1, 1.3), (1.8, 2.0), (2.3, 2.6)],
 "Qwen2.5-1.5B-Instruct": [(1.1, 1.4), (1.8, 2.1), (2.6, 3.0)],
 "Mistral-7B-v0.1": [(0.6, 0.8), (0.9, 1.0), (1.3, 1.4)],
}
h = 0.0005
for m, rows in T1.items():
    d = rec(RM, f"agreement_repeats_{m}.json")
    w = f"Table 1 {m}"
    check(f"{w} sequences", SEQS[m][0], d["n_seqs"], 0)
    check(f"{w} positions", SEQS[m][1], d["n_positions"], 0)
    check(f"{w} bootstrap resamples", 2000, d["bootstrap"]["n"], 0)
    assert d["bootstrap"]["unit"] == "sequence"
    check(f"{w} base accuracy", rows["base"], d["quantized"]["8"]["acc_base"], h)
    for b in ["8", "4", "3"]:
        q, p = d["quantized"][b], rows[b]
        check(f"{w} {b}-bit accuracy", p[0], q["acc_quant"], h)
        check(f"{w} {b}-bit dAcc", p[1], q["acc_drop"], h + 1e-4)
        check(f"{w} {b}-bit dAcc lo", p[2][0], q["acc_drop_ci95"][0], h)
        check(f"{w} {b}-bit dAcc hi", p[2][1], q["acc_drop_ci95"][1], h)
        check(f"{w} {b}-bit r", p[3], q["regressions"], h)
        check(f"{w} {b}-bit c", p[4], q["recoveries"], h)
        check(f"{w} {b}-bit 1-A", p[5], q["disagreement"], h)
        check(f"{w} {b}-bit 1-A lo", p[6][0], q["disagreement_ci95"][0], h)
        check(f"{w} {b}-bit 1-A hi", p[6][1], q["disagreement_ci95"][1], h)
        check(f"{w} {b}-bit D-D0", p[7], q["d_minus_d0_ref"], h)
        check(f"{w} {b}-bit D-D0 lo", p[8][0], q["d_minus_d0_ref_ci95"][0], h)
        check(f"{w} {b}-bit D-D0 hi", p[8][1], q["d_minus_d0_ref_ci95"][1], h)
        # the identities the Lean file proves, on the recorded counts
        assert abs((q["regressions"] - q["recoveries"]) - q["acc_drop"]) < 1e-9
        assert abs(q["ca"] - (q["acc_base"] - q["regressions"])) < 1e-9
    for (lo, hi), amp in zip(D0[m], ["0.02", "0.05", "0.1"]):
        v = d["d0_by_amplitude"][amp]
        check(f"{w} D0 range low at dither {amp} (%)", lo, 100 * min(v), 0.05)
        check(f"{w} D0 range high at dither {amp} (%)", hi, 100 * max(v), 0.05)
q = rec(RM, "agreement_repeats_Llama-3.2-3B.json")["quantized"]["8"]
check("Table 1 text Llama 8-bit Correctness Agreement", 0.493, q["ca"], h)
check("Table 1 text Llama 8-bit regressions (%)", 0.33, 100 * q["regressions"], 0.005)
check("Table 1 text Llama 8-bit recoveries (%)", 0.33, 100 * q["recoveries"], 0.005)
report("Table 1")
''')

md("""
## Tables 2, 4 and 7 and the earlier figures of Sections 4 and 7

These come from the first set of records. The paper's two earlier scripts,
`scripts/derive_numbers.py` and `scripts/derive_ratios.py`, print them. The cell below runs both
scripts on copies of the records and compares their output with the paper.
""")

code(r'''
tmp = tempfile.mkdtemp(prefix="tmlr_records_")
names = ([f"behavioral_{m}.json" for m in ["Qwen2.5-1.5B-Instruct", "Mistral-7B-v0.1", "Llama-3.2-3B"]]
         + [f"matched_bit_{m}.json" for m in ["Qwen2.5-1.5B-Instruct", "gemma-3-4b-it", "Mistral-7B-v0.1", "Llama-3.2-3B"]]
         + [f"k_wavelength_{m}.json" for m in ["Llama-3.2-3B", "Mistral-7B-v0.1"]]
         + ["validate_rope_aware_k_Llama-3.2-3B.json"])
for n in names:
    shutil.copy(os.path.join(RM, n), tmp)
for n in ["results_mixtral_routing.json", "results_olmoe_routing.json", "results_mamba_decay.json"]:
    shutil.copy(os.path.join(BM, n), tmp)
out = ""
for script in ["derive_numbers.py", "derive_ratios.py"]:
    r = subprocess.run([sys.executable, "-B", os.path.join(PAPER, "scripts", script)], cwd=tmp,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    out += ("== three-decimal ratios\n" if script == "derive_ratios.py" else "") + r.stdout
shutil.rmtree(tmp, ignore_errors=True)
print(out[-1400:])

def block(model):
    """The lines of `out` for one model, per section, keyed by the section header."""
    res, sec = {}, None
    for line in out.splitlines():
        if line.startswith("== "):
            sec = line
        elif sec is not None and not line.startswith(" "):
            cur = line.split(" ")[0]
            res.setdefault((sec, cur), [line])
        elif sec is not None and res:
            res[list(res)[-1]].append(line)
    return {k: "\n".join(v) for k, v in res.items() if model in k[1]}

def num(pattern, text):
    m = re.search(pattern, text)
    assert m, pattern
    return float(m.group(1))

# Table 2: functional KL ratio QK / VO, and the range of the relative Frobenius ratio
T2 = {"Qwen/Qwen2.5-1.5B-Instruct": ((0.94, 1.00), {"8": 0.38, "4": 0.26, "3": 0.17, "2": 0.27}),
      "unsloth/gemma-3-4b-it": ((1.03, 1.04), {"8": 0.83, "4": 0.43, "3": 0.43}),
      "mistralai/Mistral-7B-v0.1": ((1.01, 1.13), {"8": 0.29, "4": 0.29, "3": 0.44, "2": 1.05}),
      "unsloth/Llama-3.2-3B": ((1.00, 1.05), {"8": 0.40, "4": 0.36, "3": 2.77, "2": 1.13})}
INV4 = {"Qwen/Qwen2.5-1.5B-Instruct": 3.78, "unsloth/gemma-3-4b-it": 2.35,
        "mistralai/Mistral-7B-v0.1": 3.41, "unsloth/Llama-3.2-3B": 2.75}
ratio_lines = out[out.index("== three-decimal ratios"):].splitlines()
for m, (frange, kl) in T2.items():
    line = [l for l in ratio_lines if l.startswith(m)][0]
    wf = []
    for b, printed in kl.items():
        seg = re.search(rf"b{b} wf=([\d.]+) ok=([\d.]+) inv=([\d.]+)", line)
        wf.append(float(seg.group(1)))
        check(f"Table 2 {m} {b}-bit KL ratio", printed, float(seg.group(2)), 0.005)
        if b == "4":
            check(f"Table 2 text {m} 4-bit VO/QK", INV4[m], float(seg.group(3)), 0.005)
    check(f"Table 2 {m} Frobenius ratio, low end", frange[0], min(wf), 0.005)
    check(f"Table 2 {m} Frobenius ratio, high end", frange[1], max(wf), 0.005)

mb = out[out.index("== matched-bit"):out.index("== k-wavelength")]
mis = mb[mb.index("mistralai/Mistral-7B-v0.1"):]
b4 = [l for l in mis.splitlines() if l.strip().startswith("b4")][0]
for name, printed in [("Q", 0.034), ("K", 0.036), ("V", 0.075), ("O", 0.048)]:
    check(f"Table 2 text Mistral 4-bit flip rate {name}", printed, num(rf"flip .*?{name}=([\d.]+)", b4), 0.0005)
lla = mb[mb.index("unsloth/Llama-3.2-3B"):mb.index("mistralai/Mistral-7B-v0.1")]
b3 = [l for l in lla.splitlines() if l.strip().startswith("b3")][0]
for name, printed in [("Q", 0.089), ("K", 0.741), ("V", 0.125), ("O", 0.174)]:
    check(f"Section 4.2 Llama 3-bit KL {name}", printed, num(rf"kl .*?{name}=([\d.e+-]+)", b3), 0.0005)
for name, printed in [("Q", 0.22), ("K", 1.71), ("V", 0.31), ("O", 0.43)]:
    check(f"Section 4.2 Llama 3-bit movement per unit weight error {name}", printed,
          num(rf"amp .*?{name}=([\d.]+)", b3), 0.005)
qw = mb[mb.index("Qwen/Qwen2.5-1.5B-Instruct"):mb.index("unsloth/gemma-3-4b-it")]
b3q = [l for l in qw.splitlines() if l.strip().startswith("b3")][0]
check("Section 4.2 Qwen 3-bit KL O", 0.624, num(r"kl .*?O=([\d.e+-]+)", b3q), 0.0005)
check("Section 4.2 Qwen 3-bit KL V", 0.183, num(r"kl .*?V=([\d.e+-]+)", b3q), 0.0005)
report("Table 2"); report("Section 4.2")

# Table 4 and the protection replicated on other hardware
kw = out[out.index("== k-wavelength"):out.index("== rope-aware")]
T3 = {"unsloth/Llama-3.2-3B": (0.741, 0.905, 0.72, 87, 0.12, 0.173),
      "mistralai/Mistral-7B-v0.1": (0.030, 0.571, 0.79, 49, 0.22, 0.031)}
for m, p in T3.items():
    seg = kw[kw.index(m):]
    seg = "\n".join(seg.splitlines()[:2])
    check(f"Table 4 {m} joint KL", p[0], num(r"full_k_kl=([\d.]+)", seg), 0.0005)
    check(f"Table 4 {m} octile Spearman", p[1], num(r"bucket_spearman=([\d.]+)", seg), 0.0005)
    check(f"Table 4 {m} two longest octiles share", p[2], num(r"top2_share=([\d.]+)", seg), 0.005)
    check(f"Table 4 {m} KL removed (%)", p[3], num(r"\((\d+)% recovered\)", seg), 0.5)
    check(f"Table 4 {m} row weight error Spearman", p[4], num(r"row_spearman_relfro=([\d.]+)", seg), 0.005)
    check(f"Table 4 {m} sum of single-octile KLs", p[5], num(r"sum kl_only=([\d.]+)", seg), 0.0005)
rp = out[out.index("== rope-aware"):out.index("== routing")]
for name, pat, printed, tol in [("unprotected", r"unprot=([\d.]+) prot=", 0.7352, 5e-5),
                                ("protected", r" prot=([\d.]+) prot8", 0.0994, 5e-5),
                                ("protected at 8 bits", r"prot8=([\d.]+)", 0.0994, 5e-5),
                                ("recovery (%)", r"recovery=([\d.]+)%", 86.5, 0.05),
                                ("whole model 4-bit protected", r"full b4 prot=([\d.]+)", 0.4668, 5e-5),
                                ("whole model 4-bit unprotected", r"full b4 prot=[\d.]+ unprot=([\d.]+)", 0.4787, 5e-5),
                                ("whole model gain (%)", r"gain ([\d.]+)%", 2.5, 0.05)]:
    check(f"Table 4 text, separate implementation: {name}", printed, num(pat, rp), tol)
report("Table 4")

# Table 8: routing
rt = out[out.index("== routing"):out.index("== mamba")]
T5 = {"mistralai/Mixtral-8x7B-Instruct-v0.1": dict(k=2, dec=131072, p50=0.485, p10=0.076, g4=(16.6, 10.7),
                                                 g3=(34.0, 2.69), sweep=[(9.2, 2007), (17.9, 16.7), (34.1, 3.4), (58.3, 1.6)]),
      "allenai/OLMoE-1B-7B-0924": dict(k=8, dec=262144, p50=0.0015, p10=0.0002, g4=(91.8, 1.09),
                                       g3=(98.7, 1.01), sweep=[(10.6, 1740), (20.2, 22.6), (35.7, 4.6), (57.7, 2.1)])}
for m, p in T5.items():
    seg = rt[rt.index(m):]
    seg = "\n".join(seg.splitlines()[:5])
    w = f"Table 8 {m.split('/')[1]}"
    check(f"{w} decisions", p["dec"], num(r"decisions=(\d+)", seg), 0)
    check(f"{w} median margin", p["p50"], num(r"p50=([\d.]+)", seg), 0.0005 if p["p50"] > 0.01 else 5e-5)
    check(f"{w} 10th percentile margin", p["p10"], num(r"p10=([\d.]+)", seg), 0.0005 if p["p10"] > 0.01 else 5e-5)
    for bits, key in [("4", "g4"), ("3", "g3")]:
        check(f"{w} gate {bits}-bit flips (%)", p[key][0], 100 * num(rf"gate {bits}-bit: flip=([\d.]+)", seg), 0.05)
        check(f"{w} gate {bits}-bit low/high", p[key][1], num(rf"gate {bits}-bit: .*?ratio=([\d.]+)", seg),
              0.05 if p[key][1] > 5 else 0.005)
    sw = [l for l in seg.splitlines() if f"sweep k{p['k']}:" in l][0]
    for (flip, ratio), s_ in zip(p["sweep"], ["0.25", "0.50", "1.00", "2.00"]):
        mm = re.search(rf"s{s_}:flip=([\d.]+),ratio=([\d.]+)", sw)
        check(f"{w} noise {s_} flips (%)", flip, 100 * float(mm.group(1)), 0.05)
        check(f"{w} noise {s_} low/high", ratio, float(mm.group(2)), 0.5 if ratio > 100 else 0.05)
report("Table 8")

mam = out[out.index("== mamba"):out.index("== sha256")]
check("Section 7 baseline perplexity", 11.65, num(r"base=([\d.]+)", mam), 0.005)
check("Section 7 log grid perplexity", 14.44, num(r"log_tau=([\d.]+)", mam), 0.005)
check("Section 7 linear grid perplexity, first implementation (1e10)", 1.0, num(r"linear=([\d.e+]+)", mam) / 1e10, 0.05)
check("Section 7 largest steady-state gain", 42690, num(r"max_gain=(\d+)", mam), 0.5)
check("Section 7 channels", 2359296, num(r"n_ch=(\d+)", mam), 0)
''')

md("""
## Table 3: where the two orderings part (the llama.cpp factorial)

Record: `experiments/results_matched_bit/llamacpp_factorial_<model>.json`.
""")

code(r'''
ARMS = ["kquant_mix", "kquant_pure", "chan_mix", "chan_equal"]
TF = {  # (rel. Frobenius, |d kurtosis|, functional KL), rows in the order of ARMS
 ("Llama-3.2-3B", "Q2_K"): [(1.97, 4.24, 0.91), (1.00, 1.23, 0.17), (2.27, 333, 31.6), (1.00, 1.25, 1.13)],
 ("Llama-3.2-3B", "Q3_K_M"): [(2.17, 1.94, 1.12), (1.01, 1.20, 0.23), (2.42, 3.62, 15.3), (1.04, 1.52, 2.77)],
 ("Mistral-7B-v0.1", "Q2_K"): [(2.72, 6.25, 1.18), (1.02, 1.92, 0.057), (3.13, 205, 67.3), (1.01, 1.47, 1.05)],
 ("Mistral-7B-v0.1", "Q3_K_M"): [(2.16, 2.94, 0.61), (1.01, 1.02, 0.17), (2.54, 4.08, 2.47), (1.09, 1.57, 0.45)],
}
def tol_of(x):
    s_ = repr(x)
    dec = len(s_.split(".")[1]) if "." in s_ else 0
    return 0.5 * 10 ** (-dec) + 1e-12
fr = {}
for (m, level), rows in TF.items():
    d = rec(RM, f"llamacpp_factorial_{m}.json")["results"][level]
    fr[(m, level)] = d
    for arm, p in zip(ARMS, rows):
        a = d["arms"][arm]
        w = f"Table 3 {m} {level} {arm}"
        check(f"{w} rel. Frobenius", p[0], a["weight_rel_fro_qk_over_vo"], 0.005)
        check(f"{w} |d kurtosis|", p[1], a["weight_d_kurtosis_qk_over_vo"], tol_of(p[1]))
        check(f"{w} functional KL", p[2], a["functional_qk_over_vo"], tol_of(p[2]))
# the shipped tensor types quoted in the text
assert fr[("Llama-3.2-3B", "Q2_K")]["types_mix"] == {"Q": ["Q2_K"], "K": ["Q2_K"], "V": ["Q3_K"], "O": ["Q3_K"]}
assert fr[("Mistral-7B-v0.1", "Q2_K")]["types_mix"] == {"Q": ["Q2_K"], "K": ["Q2_K"], "V": ["Q4_K"], "O": ["Q3_K"]}
for m in ["Llama-3.2-3B", "Mistral-7B-v0.1"]:
    t = fr[(m, "Q3_K_M")]["types_mix"]
    assert t["Q"] == ["Q3_K"] and t["K"] == ["Q3_K"] and t["O"] == ["Q4_K"] and set(t["V"]) == {"Q4_K", "Q5_K"}
# ranges quoted in the text
allr = lambda arm, key: [fr[k]["arms"][arm][key] for k in TF]
check("Table 3 text shipped mix Frobenius low", 2.0, min(allr("kquant_mix", "weight_rel_fro_qk_over_vo")), 0.05)
check("Table 3 text shipped mix Frobenius high", 2.7, max(allr("kquant_mix", "weight_rel_fro_qk_over_vo")), 0.05)
check("Table 3 text shipped mix kurtosis low", 1.9, min(allr("kquant_mix", "weight_d_kurtosis_qk_over_vo")), 0.05)
check("Table 3 text shipped mix kurtosis high", 6.3, max(allr("kquant_mix", "weight_d_kurtosis_qk_over_vo")), 0.05)
check("Table 3 text equal type Frobenius low", 1.00, min(allr("kquant_pure", "weight_rel_fro_qk_over_vo")), 0.005)
check("Table 3 text equal type Frobenius high", 1.02, max(allr("kquant_pure", "weight_rel_fro_qk_over_vo")), 0.005)
check("Table 3 text equal type kurtosis low", 1.0, min(allr("kquant_pure", "weight_d_kurtosis_qk_over_vo")), 0.05)
check("Table 3 text equal type kurtosis high", 1.9, max(allr("kquant_pure", "weight_d_kurtosis_qk_over_vo")), 0.05)
check("Table 3 text per-channel shipped bits Frobenius low", 2.3, min(allr("chan_mix", "weight_rel_fro_qk_over_vo")), 0.05)
check("Table 3 text per-channel shipped bits Frobenius high", 3.1, max(allr("chan_mix", "weight_rel_fro_qk_over_vo")), 0.05)
check("Table 3 text per-channel equal bits Frobenius low", 1.00, min(allr("chan_equal", "weight_rel_fro_qk_over_vo")), 0.005)
check("Table 3 text per-channel equal bits Frobenius high", 1.09, max(allr("chan_equal", "weight_rel_fro_qk_over_vo")), 0.005)
inv = [1 / v for v in allr("kquant_pure", "functional_qk_over_vo")]
check("Table 3 text VO over QK at equal type, low (4 times)", 4, min(inv), 0.5)
check("Table 3 text VO over QK at equal type, high (17 times)", 17, max(inv), 0.5)
check("Table 3 text shipped mix functional low", 0.61, min(allr("kquant_mix", "functional_qk_over_vo")), 0.005)
check("Table 3 text shipped mix functional high", 1.18, max(allr("kquant_mix", "functional_qk_over_vo")), 0.005)
kl2c = [fr[(m, "Q2_K")]["arms"]["chan_equal"][p]["functional"]["kl"] for m in ["Llama-3.2-3B", "Mistral-7B-v0.1"] for p in "QKVO"]
kl2k = [fr[(m, "Q2_K")]["arms"]["kquant_pure"][p]["functional"]["kl"] for m in ["Llama-3.2-3B", "Mistral-7B-v0.1"] for p in "QKVO"]
check("Table 3 text per-channel 2-bit KL low", 2.4, min(kl2c), 0.05)
check("Table 3 text per-channel 2-bit KL high", 5.5, max(kl2c), 0.05)
check("Table 3 text K-quant 2-bit KL low", 0.006, min(kl2k), 0.0005)
check("Table 3 text K-quant 2-bit KL high", 0.23, max(kl2k), 0.005)
report("Table 3")
''')

md("""
## Score-error decomposition (Section 4.1, "a partial test")

Record: `experiments/results_matched_bit/score_error_<model>.json`.
""")

code(r'''
se = {m: rec(RM, f"score_error_{m}.json")["results"]
      for m in ["Qwen2.5-1.5B-Instruct", "gemma-3-4b-it", "Mistral-7B-v0.1", "Llama-3.2-3B"]}
col = lambda key, field: [se[m][key][field] for m in se]
check("Score error: row-constant share, query, 4 bits, low", 0.47, min(col("Q4", "const_share_layer_median")), 0.005)
check("Score error: row-constant share, query, 4 bits, high", 0.52, max(col("Q4", "const_share_layer_median")), 0.005)
check("Score error: row-constant share, key, 4 bits, low", 0.22, min(col("K4", "const_share_layer_median")), 0.005)
check("Score error: row-constant share, key, 4 bits, high", 0.33, max(col("K4", "const_share_layer_median")), 0.005)
c4 = col("Q4", "centered_rel") + col("K4", "centered_rel")
c3 = col("Q3", "centered_rel") + col("K3", "centered_rel")
check("Score error: centered error, 4 bits, low (%)", 6, 100 * min(c4), 0.5)
check("Score error: centered error, 4 bits, high (%)", 11, 100 * max(c4), 0.5)
check("Score error: centered error, 3 bits, low (%)", 15, 100 * min(c3), 0.5)
check("Score error: centered error, 3 bits, high (%)", 24, 100 * max(c3), 0.5)
t4 = col("Q4", "attention_tv") + col("K4", "attention_tv")
t3 = col("Q3", "attention_tv") + col("K3", "attention_tv")
check("Score error: attention total variation, 4 bits, low", 0.02, min(t4), 0.005)
check("Score error: attention total variation, 4 bits, high", 0.06, max(t4), 0.005)
check("Score error: attention total variation, 3 bits, low", 0.06, min(t3), 0.005)
check("Score error: attention total variation, 3 bits, high", 0.13, max(t3), 0.005)
for m in se:
    check(f"Score error: sequences, {m}", 8, rec(RM, f"score_error_{m}.json")["n_seqs"], 0)
report("Score error")
''')

md("""
## Table 5: row protection at a matched budget, on held-out text

Record: `experiments/results_matched_bit/k_row_protection_<model>.json`.
""")

code(r'''
TP = {  # condition: (KL, [lo, hi], removed %) per model; random conditions: (KL low, KL high, removed low, removed high)
 "Llama-3.2-3B": {"none": (0.781, (0.715, 0.837), None), "shortest": (0.749, (0.688, 0.812), 4),
                  "longest": (0.089, (0.083, 0.095), 89), "act_error": (0.031, (0.030, 0.033), 96),
                  "weight_error": (0.028, (0.027, 0.029), 96),
                  "random_rows": (0.491, 0.732, 6, 37), "random_freqs": (0.350, 0.627, 20, 55)},
 "Mistral-7B-v0.1": {"none": (0.0279, (0.0263, 0.0297), None), "shortest": (0.0262, (0.0248, 0.0277), 6),
                     "longest": (0.0137, (0.0128, 0.0147), 51), "act_error": (0.0095, (0.0090, 0.0101), 66),
                     "weight_error": (0.0073, (0.0069, 0.0078), 74),
                     "random_rows": (0.0237, 0.0248, 11, 15), "random_freqs": (0.0238, 0.0251, 10, 15)},
}
pr = {}
for m, rows in TP.items():
    d = rec(RM, f"k_row_protection_{m}.json")
    pr[m] = d
    r = d["results"]
    t = 0.0005 if m.startswith("Llama") else 0.00005
    w = f"Table 5 {m}"
    assert d["eval_split"].endswith("validation") and d["calib_split"].endswith("train")
    check(f"{w} evaluation sequences", 64, d["n_eval_seqs"], 0)
    check(f"{w} calibration sequences", 16, d["n_calib_seqs"], 0)
    check(f"{w} bits per key weight", 3.625, d["bits_per_key_weight"], 0)
    assert d["protected_rows_per_kproj"] * 8 == d["rows_per_kproj"]
    for cond, p in rows.items():
        if cond.startswith("random"):
            ks = sorted(k for k in r if k.startswith(cond))
            assert len(ks) == 5
            kls = [r[k]["kl"] for k in ks]
            rem = [100 * r[k]["recovery_vs_none"] for k in ks]
            check(f"{w} {cond} KL low", p[0], min(kls), t)
            check(f"{w} {cond} KL high", p[1], max(kls), t)
            check(f"{w} {cond} removed low (%)", p[2], min(rem), 0.5)
            check(f"{w} {cond} removed high (%)", p[3], max(rem), 0.5)
        else:
            check(f"{w} {cond} KL", p[0], r[cond]["kl"], t)
            check(f"{w} {cond} KL lo", p[1][0], r[cond]["kl_ci95"][0], t)
            check(f"{w} {cond} KL hi", p[1][1], r[cond]["kl_ci95"][1], t)
            if p[2] is not None:
                check(f"{w} {cond} removed (%)", p[2], 100 * r[cond]["recovery_vs_none"], 0.5)
L, M = pr["Llama-3.2-3B"]["results"], pr["Mistral-7B-v0.1"]["results"]
lows = [L[k]["kl_minus_longest_ci95"][0] for k in L if k.startswith("random")]
check("Table 5 text: smallest lower end, random minus longest (above 0.22)", 0.22, min(lows), 0.01)
assert min(lows) > 0.22
check("Table 5 text: weight error minus longest, lo", -0.066, L["weight_error"]["kl_minus_longest_ci95"][0], 0.0005)
check("Table 5 text: weight error minus longest, hi", -0.056, L["weight_error"]["kl_minus_longest_ci95"][1], 0.0005)
check("Table 5 text: probe set, unprotected", 0.741, L["none"]["probe_set"]["kl"], 0.0005)
check("Table 5 text: probe set, longest", 0.099, L["longest"]["probe_set"]["kl"], 0.0005)
check("Table 5 text: unprotected damage ratio Llama / Mistral", 28, L["none"]["kl"] / M["none"]["kl"], 0.5)
check("Table 5 text: overlap of weight-error rows with longest octile, Llama (%)", 23, 100 * L["weight_error"]["overlap_with_longest"], 0.5)
check("Table 5 text: the same, Mistral (%)", 43, 100 * M["weight_error"]["overlap_with_longest"], 0.5)
report("Table 5")
''')

md("""
## Table 8, new rows: router error spread over margin

Record: `benchmarks/router_error_margin_<model>.json`. The record's field
`logit_recompute_rel_err` compares recomputed logits with the router module's output, which is a
softmax, so it validates nothing and is not used here. The flip rates below are compared with
the routing run instead.
""")

code(r'''
RR = {"Mixtral-8x7B-Instruct-v0.1": {"4": ((0.40, 1.28, 8.2), 40.9, 16.6), "3": ((0.90, 2.9, 18.5), 12.5, 34.0)},
      "OLMoE-1B-7B-0924": {"4": ((7.5, 27.9, 191), 0.003, 91.8), "3": ((12.1, 48.4, 345), 0.0, 98.7)}}
rr = {}
for m, rows in RR.items():
    d = rec(BM, f"router_error_margin_{m}.json")
    rr[m] = d
    assert "per-tensor uniform grid over [min, max]" in d["quantizer"]
    for b, (q, below, flips) in rows.items():
        v = d["results"][b]
        w = f"Table 8 R {m} {b}-bit"
        for name, printed, got in zip(["10th percentile", "median", "90th percentile"], q,
                                      [v["R_p10"], v["R_median"], v["R_p90"]]):
            check(f"{w} {name}", printed, got, tol_of(printed))
        check(f"{w} tokens with R<1 (%)", below, 100 * v["frac_R_below_1"], 0.05 if below > 1 else 0.0005)
        check(f"{w} flips (%), against the routing run", flips, 100 * v["flip_rate"], 0.05)
        fb = [x["flip_rate_R_below_1"] for x in v["per_layer"] if x["flip_rate_R_below_1"] is not None]
        assert all(x == 0.0 for x in fb), "a token with R < 1 changed its route"
mx = rr["Mixtral-8x7B-Instruct-v0.1"]["results"]["4"]["per_layer"]
check("Table 8 text: Mixtral layers", 32, len(mx), 0)
check("Table 8 text: Mixtral layer share with R<1, low (%)", 27, 100 * min(x["frac_R_below_1"] for x in mx), 0.5)
check("Table 8 text: Mixtral layer share with R<1, high (%)", 73, 100 * max(x["frac_R_below_1"] for x in mx), 0.5)
report("Table 8 R"); report("Table 8 text")
''')

md("""
## Section 7: what each decay grid does to the channels

Record: `benchmarks/results_mamba_decay_census.json`.
""")

code(r'''
c = rec(BM, "results_mamba_decay_census.json")
g = c["grids"]
check("Section 7 census: channels", 2359296, c["n_channels"], 0)
check("Section 7 census: baseline perplexity", 11.65, c["ppl_baseline"], 0.005)
check("Section 7 census: smallest decay rate (1e-5)", 2, c["rate_abs_A"]["min"] / 1e-5, 0.5)
check("Section 7 census: largest decay rate (1e6)", 2.9, c["rate_abs_A"]["max"] / 1e6, 0.05)
check("Section 7 census: median decay rate", 9.6, c["rate_abs_A"]["median"], 0.05)
check("Section 7 census: linear step over median rate, typical layer", 383, c["linear_step_over_median_rate"]["median"], 0.5)
assert c["linear_step_over_median_rate"]["min"] >= 38
check("Section 7 census: linear grid top-level share (%)", 99.8, 100 * g["linear"]["top_level_fraction"], 0.05)
assert 100 * g["linear"]["top_level_fraction_by_layer"]["min"] >= 99.0
check("Section 7 census: linear grid, slower by 10x (%)", 94, 100 * g["linear"]["rate_ratio_all"]["frac_slower_10x"], 0.5)
check("Section 7 census: linear grid, median slowdown (about 200)", 200, 1 / g["linear"]["rate_ratio_all"]["median"], 15)
check("Section 7 census: linear grid perplexity, second implementation (1e9)", 5.9, g["linear"]["ppl"] / 1e9, 0.05)
assert g["linear"]["ppl"] > 1e9
check("Section 7 census: log grid perplexity", 14.44, g["log_tau"]["ppl"], 0.005)
check("Section 7 census: log grid median rate ratio", 0.99, g["log_tau"]["rate_ratio_all"]["median"], 0.005)
check("Section 7 census: log grid, slower by 10x (%)", 0, 100 * g["log_tau"]["rate_ratio_all"]["frac_slower_10x"], 0.05)
check("Section 7 census: log grid, changed by 2x either way (%)", 28, 100 * g["log_tau"]["rate_ratio_all"]["frac_changed_2x"], 0.5)
check("Section 7 census: per-row linear grid top-level share (%)", 43.5, 100 * g["linear_per_row"]["top_level_fraction"], 0.05)
check("Section 7 census: per-row linear grid, slower by 10x (%)", 9, 100 * g["linear_per_row"]["rate_ratio_all"]["frac_slower_10x"], 0.5)
check("Section 7 census: per-row linear grid perplexity (1e6)", 4.1, g["linear_per_row"]["ppl"] / 1e6, 0.05)
report("Section 7")
''')

md("""
## Section 7, second model: Mamba-370m

Record: `benchmarks/results_mamba_decay_census_mamba-370m.json`.
""")

code(r'''
c2 = rec(BM, "results_mamba_decay_census_mamba-370m.json")
g2 = c2["grids"]
assert c2["model"] == "state-spaces/mamba-370m-hf" and c2["bits"] == 3
check("Section 7 second model: layers", 48, c2["n_layers"], 0)
check("Section 7 second model: channels", 1572864, c2["n_channels"], 0)
check("Section 7 second model: baseline perplexity", 13.99, c2["ppl_baseline"], 0.005)
check("Section 7 second model: log grid perplexity", 18.94, g2["log_tau"]["ppl"], 0.005)
check("Section 7 second model: linear grid top-level share (%)", 99.8, 100 * g2["linear"]["top_level_fraction"], 0.05)
check("Section 7 second model: linear grid, slower by 10x (%)", 96, 100 * g2["linear"]["rate_ratio_all"]["frac_slower_10x"], 0.5)
assert g2["linear"]["ppl"] > 1e11
check("Section 7 second model: per-row linear grid top-level share (%)", 51, 100 * g2["linear_per_row"]["top_level_fraction"], 0.5)
check("Section 7 second model: per-row linear grid perplexity (1e7)", 5.1, g2["linear_per_row"]["ppl"] / 1e7, 0.05)
check("Section 7 second model: log grid, slower by 10x (%)", 0, 100 * g2["log_tau"]["rate_ratio_all"]["frac_slower_10x"], 0.05)
check("Section 7 second model: log grid, changed by 2x (%)", 35, 100 * g2["log_tau"]["rate_ratio_all"]["frac_changed_2x"], 0.5)
check("Section 7 second model: log grid cost, perplexity ratio", 1.35, g2["log_tau"]["ppl"] / c2["ppl_baseline"], 0.005)
check("Section 7 second model: the same ratio on the first model", 1.24, g["log_tau"]["ppl"] / c["ppl_baseline"], 0.005)
report("Section 7 second model")
''')

md("""
## The linear law for selections (Section 3 and Section 6)

Prediction: flips = p(0) sigma / sqrt(pi), with p(0) estimated as 0.1 / (10th percentile of the
margin). Records: `benchmarks/results_mixtral_routing.json`, `benchmarks/results_olmoe_routing.json`.
""")

code(r'''
LAW = {  # (record, boundary): (predicted %, measured %, measured ratio when the noise doubles)
 ("results_mixtral_routing.json", "topk"): (9.0, 9.2, 1.95),
 ("results_olmoe_routing.json", "topk"): (10.5, 10.6, 1.90),
 ("results_mixtral_routing.json", "top1"): (8.2, 8.4, 2.00),
 ("results_olmoe_routing.json", "top1"): (9.7, 9.6, 1.89),
}
for (name, which), (pred, meas, ratio) in LAW.items():
    d = rec(BM, name)
    ms = d["margin_summary_" + which]
    sweep = d["differential_noise_sweep"]["k" + str(ms["k"])]
    lv = sweep["levels"]
    assert lv[0]["sigma_over_median_margin"] == 0.25 and lv[1]["sigma_over_median_margin"] == 0.5
    sigma = 0.25 * ms["margin_p50"]
    p0 = 0.1 / ms["margin_p10"]
    w = f"Linear law {name.split('_')[1]} {which}"
    check(f"{w} predicted flips (%)", pred, 100 * p0 * sigma / math.sqrt(math.pi), 0.05)
    check(f"{w} measured flips (%)", meas, 100 * lv[0]["overall_flip_rate"], 0.05)
    check(f"{w} ratio when the noise doubles", ratio, lv[1]["overall_flip_rate"] / lv[0]["overall_flip_rate"], 0.005)
    assert sigma > ms["margin_p10"], "the text says sigma exceeds the 10th percentile margin"
report("Linear law")
''')

md("""
## Table 6: the tracer on models with known consuming operators

Record: `experiments/results_matched_bit/tracer_accuracy.json`, written by
`experiments/tracer_accuracy.py` (CPU, random weights, no checkpoint).
""")

code(r'''
tr = rec(RM, "tracer_accuracy.json")
A = tr["group_a"]["auto"]
check("Table 6 library architectures: models", 8, A["pooled"]["models"], 0)
check("Table 6 library architectures: tensors scored", 200, A["pooled"]["scored"], 0)
check("Table 6 library architectures: correct", 200, A["pooled"]["correct"], 0)
check("Table 6 library architectures: symbolic trace succeeds", 0, A["pooled"]["models_where_symbolic_trace_ok"], 0)
assert not tr["group_a_skipped"], tr["group_a_skipped"]
assert set(A["per_model"]) == {"Llama", "Mistral", "Qwen2", "Gemma", "GPT-2", "Mixtral", "OLMoE", "Mamba"}
for m in ["Mixtral", "OLMoE"]:
    g_ = A["per_model"][m]["accuracy_by_true_regime"]["gate_selection"]
    assert g_["correct"] == g_["n"] > 0
B = tr["group_b"]
check("Table 6 small modules, usual names, both passes: scored", 13, B["auto"]["pooled_usual_names"]["scored"], 0)
check("Table 6 small modules, usual names, both passes: correct", 12, B["auto"]["pooled_usual_names"]["correct"], 0)
check("Table 6 small modules, uninformative names, both passes: scored", 13, B["auto"]["pooled_uninformative_names"]["scored"], 0)
check("Table 6 small modules, uninformative names, both passes: correct", 12, B["auto"]["pooled_uninformative_names"]["correct"], 0)
check("Table 6 small modules, uninformative names, names only: correct", 9, B["structural"]["pooled_uninformative_names"]["correct"], 0)
check("Table 6 small modules: symbolic trace succeeds", 4, B["auto"]["pooled_uninformative_names"]["models_where_symbolic_trace_ok"], 0)
rec_blind = B["auto"]["per_model"]["diagonal recurrence, uninformative names"]["confusion_true_by_predicted"]
assert rec_blind["linear_residual"] == {"state_decay": 1}, "the one graph-pass error is the map feeding the scan"
rec_named = B["structural"]["per_model"]["diagonal recurrence, usual names"]["confusion_true_by_predicted"]
assert rec_named["linear_residual"] == {"softmax_score": 1}, "the in_proj name rule"
gpt = A["per_model"]["GPT-2"]["mixed_truth"]
assert len(gpt) == 2 and all(x["predicted"] == "softmax_score" for x in gpt)
report("Table 6")
''')

md("""
## Formal checks

`paper/tmlr_operator/lean/OperatorChecks.lean` states the identities the text relies on, in Lean 4
with Mathlib. The cell below prints the build record and verifies that the file on disk is the one
that was built. It does not run Lean.
""")

code(r'''
import hashlib
lean = os.path.join(PAPER, "lean", "OperatorChecks.lean")
record = open(os.path.join(PAPER, "lean", "BUILD-RECORD.txt"), encoding="utf-8").read()
src = open(lean, "rb").read().replace(bytes([13, 10]), bytes([10]))
digest = hashlib.sha256(src).hexdigest()
print(record)
assert digest in record, "the Lean file differs from the one in the build record"
text = src.decode("utf-8")
print("theorems in the file:", len(re.findall(r"^theorem ", text, flags=re.M)))
print("occurrences of the word sorry:", text.count("sorry"))
assert text.count("sorry") == 0
''')

md("""
## Summary
""")

code(r'''
bad = [c for c in CHECKS if not c[3]]
print(f"{len(CHECKS) - len(bad)} of {len(CHECKS)} comparisons with the paper pass")
for where, printed, got, ok in bad:
    print("FAIL", where, "paper", printed, "record", got)
assert not bad
''')

md("""
## Rerunning the experiments (GPU)

The records above were produced by these scripts. Each needs a GPU and the model weights; they
are not run by this notebook. Set the environment variables shown in each script's docstring.

| Result | Script |
|---|---|
| Table 1 | `experiments/agreement_reference_repeats.py` |
| Table 2 | `experiments/matched_bit_projection_sensitivity.py` |
| Table 3 | `experiments/llamacpp_factorial.py` (needs GGUF files made by llama.cpp) |
| Score-error decomposition | `experiments/score_error_decomposition.py` |
| Table 4 | `experiments/k_wavelength_probe.py` |
| Table 5 | `experiments/k_row_protection_controls.py` |
| Table 6 | `experiments/tracer_accuracy.py` (CPU) |
| Table 8 | the routing harness, and `experiments/router_error_margin.py` |
| Section 7 | `benchmarks/validate_mamba_decay.py`, `benchmarks/mamba_decay_grid_census.py` (CPU; set `NB_MODEL` for the second checkpoint) |
| Figure 1 | `paper/tmlr_operator/scripts/make_fig_strata.py` (a diagram; reads no data) |
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "paper/tmlr_operator/reproduce.ipynb")
print("wrote paper/tmlr_operator/reproduce.ipynb with", len(cells), "cells")
