"""Print every number the third draft takes from experiments E1 to E6.

Run from the repository root:
    python paper/tnnls_operator/scripts/derive_e_numbers.py > paper/tnnls_operator/scripts/numbers_derived_e.txt

Reads the JSON records in experiments/results_matched_bit/ and benchmarks/ and prints
them with the sha256 prefix of each record. Nothing is recomputed from raw model
outputs; intervals are the records' own sequence bootstraps.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st

RM = "experiments/results_matched_bit/"


def load(path):
    # hash the LF form, so a Windows checkout with autocrlf prints the same prefix as the blob
    raw = open(path, "rb").read().replace(bytes([13, 10]), bytes([10]))
    print(f"\n# {path}  sha256 {hashlib.sha256(raw).hexdigest()[:12]}")
    return json.loads(raw)


def f(x, n=4):
    return f"{x:.{n}f}"


def ci(v, n=4):
    return f"[{v[0]:.{n}f}, {v[1]:.{n}f}]"


print("== E1: decision-level agreement, sequence bootstrap (2000 resamples of whole sequences)")
for m in ["Qwen2.5-1.5B-Instruct", "Mistral-7B-v0.1", "Llama-3.2-3B"]:
    d = load(RM + f"agreement_repeats_{m}.json")
    print(f"{m}: sequences {d['n_seqs']}, positions {d['n_positions']}, bootstrap {d['bootstrap']}")
    print(f"  base accuracy {f(d['quantized']['8']['acc_base'])} {ci(d['quantized']['8']['acc_base_ci95'])}")
    for b in ["8", "4", "3"]:
        q = d["quantized"][b]
        print(f"  {b} bits: acc {f(q['acc_quant'])}, drop {f(q['acc_drop'])} {ci(q['acc_drop_ci95'])}, "
              f"CA {f(q['ca'])}, r {f(q['regressions'])} {ci(q['regressions_ci95'])}, "
              f"c {f(q['recoveries'])} {ci(q['recoveries_ci95'])}, A {f(q['agreement'])}, "
              f"D=1-A {f(q['disagreement'])} {ci(q['disagreement_ci95'])}, "
              f"D-D0 {f(q['d_minus_d0_ref'])} {ci(q['d_minus_d0_ref_ci95'])}")
    for amp, vals in d["d0_by_amplitude"].items():
        print(f"  D0 at dither {amp} of a step, three seed pairs: {[f(v) for v in vals]}")
    print(f"  D0 of the reference pair {d['reference_ref0']}: {f(d['references'][d['reference_ref0']]['d0'])} "
          f"{ci(d['references'][d['reference_ref0']]['d0_ci95'])}")

print("\n== E2: row protection at 3.625 bits per key weight, held-out validation split")
for m in ["Llama-3.2-3B", "Mistral-7B-v0.1"]:
    d = load(RM + f"k_row_protection_{m}.json")
    print(f"{m}: eval {d['eval_split']}, {d['n_eval_seqs']} sequences; calibration {d['calib_split']}, "
          f"{d['n_calib_seqs']} sequences; {d['protected_rows_per_kproj']} of {d['rows_per_kproj']} rows "
          f"in each of {d['n_kproj']} key projections; {d['bits_per_key_weight']} bits")
    r = d["results"]
    for k in ["none", "longest", "shortest", "weight_error", "act_error"]:
        v = r[k]
        extra = f", minus longest {ci(v['kl_minus_longest_ci95'])}" if "kl_minus_longest_ci95" in v else ""
        print(f"  {k:13s} KL {f(v['kl'])} {ci(v['kl_ci95'])}, flips {f(v['flip'])}, "
              f"recovery {100 * v['recovery_vs_none']:.1f}%, overlap with longest {f(v['overlap_with_longest'], 3)}{extra}")
    for fam in ["random_rows", "random_freqs"]:
        ks = sorted(k for k in r if k.startswith(fam))
        kls = [r[k]["kl"] for k in ks]
        rec = [100 * r[k]["recovery_vs_none"] for k in ks]
        lo = min(r[k]["kl_minus_longest_ci95"][0] for k in ks)
        print(f"  {fam:13s} {len(ks)} seeds: KL {min(kls):.4f} to {max(kls):.4f} (mean {st.mean(kls):.4f}), "
              f"recovery {min(rec):.1f}% to {max(rec):.1f}% (mean {st.mean(rec):.1f}%), "
              f"smallest lower end of the paired interval against longest {lo:.4f}")
    for k in ["none", "longest"]:
        p = r[k].get("probe_set")
        if p:
            print(f"  {k} on the probe's test-split set: {json.dumps(p)[:300]}")

print("\n== E3: score-error decomposition (each layer perturbed alone, 8 sequences)")
for m in ["Qwen2.5-1.5B-Instruct", "gemma-3-4b-it", "Mistral-7B-v0.1", "Llama-3.2-3B"]:
    d = load(RM + f"score_error_{m}.json")
    for k, v in d["results"].items():
        print(f"  {m} {k}: row-constant share pooled {f(v['const_share'], 3)}, layer median "
              f"{f(v['const_share_layer_median'], 3)}; centered error / within-row score deviation "
              f"{f(v['centered_rel'], 3)}; attention total variation {f(v['attention_tv'], 3)}")

print("\n== E4: llama.cpp factorial (ratios are mean over Q and K divided by mean over V and O)")
for m in ["Llama-3.2-3B", "Mistral-7B-v0.1"]:
    d = load(RM + f"llamacpp_factorial_{m}.json")
    for level, v in d["results"].items():
        print(f"  {m} {level}: shipped tensor types {v['types_mix']}")
        for arm, a in v["arms"].items():
            kl = {p: float(f"{a[p]['functional']['kl']:.4g}") for p in "QKVO"}
            print(f"    {arm:12s} functional KL ratio {a['functional_qk_over_vo']:.3g}; weight ratios: "
                  f"rel. Frobenius {a['weight_rel_fro_qk_over_vo']:.3g}, |d kurtosis| "
                  f"{a['weight_d_kurtosis_qk_over_vo']:.3g}, 1-cos {a['weight_one_minus_cos_qk_over_vo']:.3g}, "
                  f"|d skewness| {a['weight_d_skewness_qk_over_vo']:.3g}, hist. KL "
                  f"{a['weight_hist_kl_qk_over_vo']:.3g}; KL by projection {kl}")

print("\n== E5: router error over margin, R = (max_i e_i - min_i e_i) / margin")
for name in ["Mixtral-8x7B-Instruct-v0.1", "OLMoE-1B-7B-0924"]:
    d = load(f"benchmarks/router_error_margin_{name}.json")
    print(f"{name}: top-{d['top_k']} of {d['n_experts']}, {d['n_routers']} routers, "
          f"{d['n_chunks']} x {d['tokens_per_chunk']} tokens; quantizer: {d['quantizer']}")
    for b, v in d["results"].items():
        pl = v["per_layer"]
        med = [x["R_median"] for x in pl]
        below = [x["frac_R_below_1"] for x in pl]
        fb = [x["flip_rate_R_below_1"] for x in pl if x["flip_rate_R_below_1"] is not None]
        chk = [x["logit_recompute_rel_err"] for x in pl if x.get("logit_recompute_rel_err") is not None]
        print(f"  {b} bits: R p10 {v['R_p10']:.3g}, median {v['R_median']:.3g}, p90 {v['R_p90']:.3g}; "
              f"share with R<1 {100 * v['frac_R_below_1']:.3g}%; flips {100 * v['flip_rate']:.1f}%; "
              f"layer medians of R {min(med):.3g} to {max(med):.3g}; layer shares with R<1 "
              f"{100 * min(below):.3g}% to {100 * max(below):.3g}%; largest flip rate among R<1 tokens "
              f"{max(fb) if fb else 'no such tokens'}; logit_recompute_rel_err "
              f"{min(chk):.3g} to {max(chk):.3g} (this field compares logits with the router's "
              f"softmax output, so it validates nothing; see the paper's Section 6 setup)")

print("\n== E6: Mamba-790m decay grids at 3 bits")
d = load("benchmarks/results_mamba_decay_census.json")
print(f"channels {d['n_channels']}, layers {d['n_layers']}; decay rate |A|: {d['rate_abs_A']}")
print(f"linear grid step over the layer's median rate: {d['linear_step_over_median_rate']}")
print(f"baseline perplexity {d['ppl_baseline']:.2f}")
for g, v in d["grids"].items():
    a, s = v["rate_ratio_all"], v["rate_ratio_slowest_tenth"]
    top = f"top-level share {100 * v['top_level_fraction']:.1f}% (layers {100 * v['top_level_fraction_by_layer']['min']:.1f}% " \
          f"to {100 * v['top_level_fraction_by_layer']['max']:.1f}%), " if "top_level_fraction" in v else ""
    print(f"  {g:14s} perplexity {v['ppl']:.4g}; {top}rate ratio median {a['median']:.3g}; "
          f"channels at least 2x slower {100 * a['frac_slower_2x']:.1f}%, 10x slower {100 * a['frac_slower_10x']:.1f}%, "
          f"changed 2x either way {100 * a['frac_changed_2x']:.1f}%; slowest tenth: 2x slower "
          f"{100 * s['frac_slower_2x']:.1f}%, 10x slower {100 * s['frac_slower_10x']:.1f}%")
e = load("benchmarks/results_mamba_decay.json")
print("earlier record:", json.dumps(e)[:700])
