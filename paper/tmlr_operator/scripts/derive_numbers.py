"""Re-derive the numbers the TMLR draft quotes, from the committed JSON records.

Read-only. Prints one block per result family plus the sha256 of each input.
Run from a directory that holds copies of the records listed in ../README.md.
"""

import glob
import hashlib
import json
import os

H = {}


def load(p):
    b = open(p, "rb").read()
    H[os.path.basename(p)] = hashlib.sha256(b).hexdigest()[:16]
    return json.loads(b)


def mean(xs):
    return sum(xs) / len(xs)


print("== behavioral")
for p in sorted(glob.glob("behavioral_*.json")):
    d = load(p)
    fd = d["floor"]["floor_disagreement"]
    print(
        f"{d['model']} acc_base={d['acc_base']:.3f} n={d['n_positions']} floor_dis={fd:.3f}"
    )
    for r in d["rows"]:
        print(
            f"  b{r['bits']} acc={r['acc_quant']:.3f}"
            f" dAcc={d['acc_base'] - r['acc_quant']:.3f}"
            f" CA={r['correctness_agreement']:.3f}"
            f" regr={r['regressions']:.4f} recov={r['recoveries']:.4f}"
            f" churn={r['churn']:.3f} agree={r['behavioral_agreement']:.3f}"
            f" dis={r['disagreement']:.3f} excess={r['excess_over_floor']:.3f}"
            f" z={r['excess_z']:.0f} excess/floor={r['disagreement'] / fd:.1f}x"
        )

print("== matched-bit (ratios are mean(Q,K)/mean(V,O))")
for p in sorted(glob.glob("matched_bit_*.json")):
    d = load(p)
    by = {}
    for r in d["results"]:
        by.setdefault(r["bits"], {})[r["proj"]] = r
    print(f"{d['model']} n_seq={d.get('n_seq')}")
    for b in sorted(by, reverse=True):
        x = by[b]
        if not all(k in x for k in "QKVO"):
            continue
        wf = mean([x["Q"]["w_rel_fro"], x["K"]["w_rel_fro"]]) / mean(
            [x["V"]["w_rel_fro"], x["O"]["w_rel_fro"]]
        )
        ok = mean([x["Q"]["out_kl"], x["K"]["out_kl"]]) / mean(
            [x["V"]["out_kl"], x["O"]["out_kl"]]
        )
        fl = " ".join(f"{k}={x[k]['out_top1_flip']:.3f}" for k in "QKVO")
        kl = " ".join(f"{k}={x[k]['out_kl']:.3g}" for k in "QKVO")
        am = " ".join(f"{k}={x[k]['amplification']:.2f}" for k in "QKVO")
        print(
            f"  b{b} relfro_QK/VO={wf:.2f} outkl_QK/VO={ok:.2f} (VO/QK={1 / ok:.1f}x)"
            f" | flip {fl} | kl {kl} | amp {am}"
        )

print("== k-wavelength probe")
for p in sorted(glob.glob("k_wavelength_*.json")):
    d = load(p)
    bk = d["buckets"]
    last = bk[-1]
    full = d["full_k"]["out_kl"]
    print(
        f"{d['model']} full_k_kl={full:.3f}"
        f" bucket_spearman={d['bucket_spearman_kl_only']:.3f}"
        f" top2_share={d['damage_share_top2_buckets']:.2f}"
        f" row_spearman_relfro={d['row_spearman']['relfro']:.2f}"
    )
    print(
        f"  longest octile: wavelength {last['wavelength_lo']:.3g}-"
        f"{last['wavelength_hi']:.3g} tok, spared kl={last['kl_spared']:.3f}"
        f" ({100 * (1 - last['kl_spared'] / full):.0f}% recovered);"
        f" sum kl_only={sum(b['kl_only'] for b in bk):.3f}; n_buckets={len(bk)}"
    )

print("== rope-aware K (shipped path)")
for p in sorted(glob.glob("validate_rope_aware_k_*.json")):
    d = load(p)
    r = d["results"]
    u = r["k_only_b3_unprotected"]["out_kl"]
    pr = r["k_only_b3_protected"]["out_kl"]
    p8 = r["k_only_b3_protect8bit"]["out_kl"]
    print(
        f"{d['model']} {d['provenance']['gpu']} unprot={u:.4f} prot={pr:.4f}"
        f" prot8={p8:.4f} recovery={100 * (1 - pr / u):.1f}%"
    )
    fp = r["full_b4_protected"]["out_kl"]
    fu = r["full_b4_unprotected"]["out_kl"]
    print(f"  full b4 prot={fp:.4f} unprot={fu:.4f} (gain {100 * (1 - fp / fu):.1f}%)")

print("== routing")
for p in sorted(glob.glob("results_*_routing.json")):
    d = load(p)
    m = d["margin_summary_topk"]
    print(
        f"{d['model']} top{d['top_k']} of {d['n_experts']},"
        f" decisions={d['n_routing_decisions']},"
        f" boundary margin p10={m['margin_p10']:.4f} p50={m['margin_p50']:.4f}"
    )
    for b, g in sorted(d["gate_weight_quant"].items(), reverse=True):
        print(
            f"  gate {b}-bit: flip={g['overall_flip_rate']:.3f}"
            f" low={g['low_margin_flip_rate']:.3f}"
            f" high={g['high_margin_flip_rate']:.4f}"
            f" ratio={g['low_over_high_ratio']:.2f}"
        )
    for kk, sw in d["differential_noise_sweep"].items():
        levels = " ".join(
            f"s{lv['sigma_over_median_margin']:.2f}:flip={lv['overall_flip_rate']:.3f},"
            f"ratio={lv['low_over_high_ratio']:.1f}"
            for lv in sw["levels"]
        )
        print(f"  sweep {kk}: {levels}")

print("== mamba")
for p in sorted(glob.glob("results_mamba_decay.json")):
    d = load(p)
    s = d["sensitivity"]
    print(
        f"{d['model']} bits={d['bits']} base={d['baseline_ppl']:.2f}"
        f" linear={d['ppl_linear']:.3g} log_tau={d['ppl_log_tau']:.2f}"
        f" ratio={d['ppl_ratio_linear_over_log']:.2g} max_gain={s['max_gain']:.0f}"
        f" slow_frac={s['slow_channel_fraction']:.5f} n_ch={s['n_channels']}"
        f" seq={s['seq_len']}"
    )

print("== sha256[:16]")
for k, v in sorted(H.items()):
    print(" ", v, k)
