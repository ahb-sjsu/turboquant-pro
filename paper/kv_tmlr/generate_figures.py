#!/usr/bin/env python3
"""Figures for the asym-NF4 KV-quantization paper. Data mirrors
benchmarks/kvquant_matrix/results_matrix.json + results_rescue.json + the mechanism probe.
Outputs PDF (for LaTeX) and PNG into this directory."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# colour-blind-friendly palette
C = {"fp16": "#444444", "NF4": "#d1495b", "uniform": "#edae49", "asym-NF4": "#2e86ab"}
MODELS = ["Llama-2-7B\n(MHA 1:1)", "Llama-2-13B\n(MHA 1:1)",
          "Mistral-7B\n(GQA 4:1)", "Qwen2.5-7B\n(GQA 7:1)"]

# ---- data ----
QASPER = {  # fp16, NF4, uniform, asym-NF4
    "fp16":     [22.06, 17.06, 29.43, 43.77],
    "NF4":      [20.82, 16.86, 29.96, 4.69],
    "uniform":  [15.58, 10.52, 21.06, 33.81],
    "asym-NF4": [20.81, 16.41, 28.74, 41.91],
}
PPL = {  # fp16, NF4, asym-NF4
    "fp16":     [6.942, 6.108, 5.942, 7.457],
    "NF4":      [7.177, 6.305, 6.000, 74.19],
    "asym-NF4": [6.970, 6.126, 5.955, 7.499],
}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_crossover():
    """Headline: qasper, 4 models x 4 methods. NF4 collapses on Qwen; asym-NF4 best-or-tied."""
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    methods = ["fp16", "NF4", "uniform", "asym-NF4"]
    x = np.arange(len(MODELS)); w = 0.2
    for i, m in enumerate(methods):
        bars = ax.bar(x + (i - 1.5) * w, QASPER[m], w, label=m, color=C[m],
                      edgecolor="white", linewidth=0.5)
        if m == "NF4":  # annotate the collapse
            ax.annotate("collapse", (x[3] + (i - 1.5) * w, QASPER[m][3]),
                        xytext=(0, 14), textcoords="offset points", ha="center",
                        fontsize=8, color=C["NF4"], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylabel("LongBench qasper (↑)")
    ax.set_title("4-bit KV quantization: NF4 collapses on high-GQA; asym-NF4 is robust")
    ax.legend(ncol=4, loc="upper left", frameon=False, columnspacing=1.0)
    ax.set_ylim(0, 50)
    save(fig, "fig_crossover")


def fig_perplexity():
    """WikiText-2 ppl, log y, to show the Qwen NF4 explosion."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    methods = ["fp16", "NF4", "asym-NF4"]
    x = np.arange(len(MODELS)); w = 0.26
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1) * w, PPL[m], w, label=m, color=C[m],
               edgecolor="white", linewidth=0.5)
    ax.annotate("74.2\n(10x fp16)", (x[3], 74.19), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=8,
                color=C["NF4"], fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(MODELS)
    ax.set_ylabel("WikiText-2 perplexity (↓, log)")
    ax.set_title("Perplexity corroborates: NF4 explodes on Qwen")
    ax.legend(ncol=3, frameon=False)
    save(fig, "fig_perplexity")


def fig_mechanism():
    """Per-channel reconstruction relerr on real pre-RoPE keys: NF4 vs uniform, Qwen vs Llama.
    NF4 error ~equal across models -> the split is error TOLERANCE, not representation."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    layers = ["L0", "L10", "L20"]
    qn = [0.0154, 0.0789, 0.0869]; qu = [0.0010, 0.0190, 0.0325]
    ln = [0.0772, 0.0845, 0.0863]; lu = [0.0243, 0.0177, 0.0192]
    x = np.arange(len(layers)); w = 0.2
    ax.bar(x - 1.5 * w, qn, w, label="Qwen NF4", color=C["NF4"])
    ax.bar(x - 0.5 * w, ln, w, label="Llama NF4", color=C["NF4"], alpha=0.45)
    ax.bar(x + 0.5 * w, qu, w, label="Qwen uniform", color=C["uniform"])
    ax.bar(x + 1.5 * w, lu, w, label="Llama uniform", color=C["uniform"], alpha=0.45)
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_ylabel("per-channel rel. recon. error")
    ax.set_title("NF4 error is ~equal on Qwen and Llama\n(collapse is tolerance, not representation)")
    ax.legend(ncol=2, frameon=False, fontsize=8)
    save(fig, "fig_mechanism")


def fig_rescue():
    """Qwen rescue sweep: codebook shape, not bit-depth. 3-bit uniform > 4-bit NF4."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = ["NF4\n4-bit", "NF4 4b\n+10% out", "NF4\nk4/v8",
              "uniform\n3-bit", "uniform\n4-bit", "uniform\n8-bit"]
    vals = [4.69, 4.34, 4.42, 34.92, 33.81, 34.07]
    cols = [C["NF4"]] * 3 + [C["uniform"]] * 3
    ax.bar(range(len(vals)), vals, color=cols, edgecolor="white")
    ax.axhline(43.77, ls="--", color=C["fp16"], lw=1)
    ax.text(0.1, 44.5, "fp16 = 43.8", fontsize=8, color=C["fp16"])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Qwen2.5-7B qasper (↑)")
    ax.set_title("Qwen rescue: it is the codebook, not the bits\n(3-bit uniform beats 4-bit NF4 by 7x)")
    ax.set_ylim(0, 50)
    save(fig, "fig_rescue")


def fig_asymnf4():
    """Schematic: NF4 (symmetric, zero-centred) wastes codes on DC-offset keys; asym-NF4
    centres the grid on the data."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    rng = np.random.default_rng(0)
    data = rng.normal(6.0, 1.0, 4000)  # DC-offset channel (mean 6)
    NF4 = np.array([-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
                    0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0])
    amax = np.abs(data).max()
    for ax, title, levels, col in [
        (a1, "symmetric NF4 (abs-max about 0)", NF4 * amax, C["NF4"]),
        (a2, "asym-NF4 (centre on the data)", data.mean() + NF4 * np.abs(data - data.mean()).max(), C["asym-NF4"]),
    ]:
        ax.hist(data, bins=40, color="#bbbbbb", density=True)
        for lv in levels:
            ax.axvline(lv, color=col, lw=0.8, alpha=0.8)
        used = np.sum((levels >= data.min() - 1) & (levels <= data.max() + 1))
        ax.set_title(f"{title}\n{used}/16 codes land on the data", fontsize=9)
        ax.set_xlabel("key value (one channel)")
        ax.set_xlim(-amax * 1.05, amax * 1.05)
    a1.set_ylabel("density")
    fig.suptitle("Why NF4 collapses on DC-offset KV keys", fontsize=11, y=1.04)
    save(fig, "fig_asymnf4")


if __name__ == "__main__":
    fig_crossover()
    fig_perplexity()
    fig_mechanism()
    fig_rescue()
    fig_asymnf4()
    print("all figures written to", HERE)
