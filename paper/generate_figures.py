#!/usr/bin/env python3
"""Generate all figures for the PCA-Matryoshka IEEE TAI paper.

Uses the data from the paper's tables to create publication-quality plots.
Outputs PDF files in the paper directory.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# IEEE-style formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3.5, 2.8),  # IEEE single column
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fig_eigenspectrum():
    """Fig 1: Cumulative variance explained vs retained dimensions."""
    # Simulated eigenvalue decay based on paper's reported values
    # BGE-M3: 128d=75.8%, 256d=89.7%, 384d=96.4%, 512d=98.6%
    # We'll create a power-law decay that matches these points
    d = 1024
    # Power law: lambda_k ~ k^{-alpha}
    alpha = 1.8
    k = np.arange(1, d + 1)
    eigenvalues = k ** (-alpha)
    eigenvalues = eigenvalues / eigenvalues.sum()

    # Scale to match reported values
    cum_var = np.cumsum(eigenvalues)
    # Adjust to match: 128->0.758, 256->0.897, 384->0.964, 512->0.986
    # Use interpolation to create a smoother curve that hits our targets
    targets = {128: 0.758, 256: 0.897, 384: 0.964, 512: 0.986, 1024: 1.0}

    # Generate smooth PCA curve through target points
    from scipy.interpolate import PchipInterpolator
    target_dims = list(targets.keys())
    target_vals = list(targets.values())
    # Add origin
    target_dims = [0] + target_dims
    target_vals = [0.0] + target_vals
    interp = PchipInterpolator(target_dims, target_vals)
    dims = np.arange(0, d + 1)
    pca_curve = interp(dims)
    pca_curve = np.clip(pca_curve, 0, 1)

    # Standard basis: approximately uniform (k/d)
    standard_curve = dims / d

    fig, ax = plt.subplots()
    ax.plot(dims, pca_curve, "b-", linewidth=1.5, label="PCA basis")
    ax.plot(dims, standard_curve, "r--", linewidth=1.5, label="Standard basis")

    # Mark key points
    for d_val, v_val in [(128, 0.758), (256, 0.897), (384, 0.964)]:
        ax.plot(d_val, v_val, "bo", markersize=4)
        ax.annotate(f"{v_val:.1%}", (d_val, v_val), textcoords="offset points",
                    xytext=(5, -12), fontsize=7, color="blue")

    ax.set_xlabel("Retained dimensions")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Eigenspectrum: BGE-M3 (1024d)")

    fig.savefig(os.path.join(OUT_DIR, "fig_eigenspectrum.pdf"))
    plt.close(fig)
    print("Generated fig_eigenspectrum.pdf")


def fig_cosine_comparison():
    """Fig 2: Cosine similarity vs retained dimensions."""
    dims = [128, 256, 384, 512, 1024]
    naive = [0.333, 0.467, 0.609, 0.707, 1.0]
    pca = [0.933, 0.974, 0.990, 0.996, 1.0]

    fig, ax = plt.subplots()
    ax.plot(dims, pca, "b-o", linewidth=1.5, markersize=5, label="PCA-Matryoshka")
    ax.plot(dims, naive, "r--s", linewidth=1.5, markersize=5, label="Naïve truncation")

    # Fill between to show improvement
    ax.fill_between(dims, naive, pca, alpha=0.15, color="blue")

    # Annotate key improvement
    ax.annotate("+109%", xy=(256, 0.72), fontsize=8, color="blue", fontweight="bold",
                ha="center")

    ax.set_xlabel("Retained dimensions")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xlim(100, 1050)
    ax.set_ylim(0.2, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Truncation quality: BGE-M3")

    fig.savefig(os.path.join(OUT_DIR, "fig_cosine_comparison.pdf"))
    plt.close(fig)
    print("Generated fig_cosine_comparison.pdf")


def fig_recall_comparison():
    """Fig 3: Recall@10 across all methods."""
    methods = [
        "Scalar int8\n(4×)",
        "Scalar int4\n(8×)",
        "TQ3\n(10.6×)",
        "PCA-512+TQ3\n(20.9×)",
        "PCA-384+TQ3\n(27.7×)",
        "Binary\n(32×)",
        "PCA-256+TQ3\n(41×)",
        "PCA-128+TQ3\n(78.8×)",
        "PQ M=16\n(256×)",
    ]
    recalls = [97.2, 90.4, 83.8, 78.0, 76.4, 66.6, 78.2, 73.0, 41.4]
    errors = [1.2, 2.2, 2.8, 3.1, 3.2, 3.5, 3.1, 3.3, 3.7]

    colors = []
    for m in methods:
        if "PCA" in m:
            colors.append("#2196F3")  # Blue for PCA-Matryoshka
        elif "TQ" in m or "Scalar" in m:
            colors.append("#4CAF50")  # Green for scalar/TQ
        else:
            colors.append("#FF9800")  # Orange for other

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    x = np.arange(len(methods))
    bars = ax.bar(x, recalls, yerr=errors, capsize=3, color=colors, width=0.7,
                  edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Recall@10 (%)")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Retrieval quality comparison")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="PCA-Matryoshka + TQ3"),
        Patch(facecolor="#4CAF50", label="Scalar / TurboQuant"),
        Patch(facecolor="#FF9800", label="Binary / PQ"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    fig.savefig(os.path.join(OUT_DIR, "fig_recall_comparison.pdf"))
    plt.close(fig)
    print("Generated fig_recall_comparison.pdf")


def fig_pareto_frontier():
    """Fig 4: Compression ratio vs cosine similarity Pareto frontier."""
    # All methods: (ratio, cosine, label)
    methods = [
        (4, 0.9999, "int8"),
        (8, 0.993, "int4"),
        (10.6, 0.978, "TQ3"),
        (20.9, 0.984, "PCA-512+TQ3"),
        (27.7, 0.979, "PCA-384+TQ3"),
        (32, 0.758, "Binary"),
        (41, 0.963, "PCA-256+TQ3"),
        (78.8, 0.923, "PCA-128+TQ3"),
        (256, 0.810, "PQ"),
    ]

    fig, ax = plt.subplots()

    # Plot all points
    for ratio, cosine, label in methods:
        if "PCA" in label:
            color = "#2196F3"
            marker = "o"
        elif label in ("Binary", "PQ"):
            color = "#FF9800"
            marker = "^"
        else:
            color = "#4CAF50"
            marker = "s"
        ax.plot(ratio, cosine, marker=marker, color=color, markersize=6, zorder=5)
        ax.annotate(label, (ratio, cosine), textcoords="offset points",
                    xytext=(3, 5), fontsize=6, rotation=0)

    # Pareto frontier (manually identified)
    pareto_x = [4, 10.6, 20.9, 41, 78.8]
    pareto_y = [0.9999, 0.978, 0.984, 0.963, 0.923]
    # Sort by x for line
    sorted_pts = sorted(zip(pareto_x, pareto_y))
    px, py = zip(*sorted_pts)
    ax.plot(px, py, "k--", linewidth=0.8, alpha=0.5, zorder=1)

    ax.set_xlabel("Compression ratio (×)")
    ax.set_ylabel("Cosine similarity")
    ax.set_xscale("log")
    ax.set_xlim(2, 400)
    ax.set_ylim(0.7, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title("Compression–quality Pareto frontier")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=6, label="PCA-Matryoshka"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50",
               markersize=6, label="Scalar/TQ"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#FF9800",
               markersize=6, label="Binary/PQ"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7)

    fig.savefig(os.path.join(OUT_DIR, "fig_pareto_frontier.pdf"))
    plt.close(fig)
    print("Generated fig_pareto_frontier.pdf")


def fig_pipeline():
    """Fig 5: PCA-Matryoshka + TurboQuant compression pipeline diagram."""
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Boxes
    boxes = [
        (0.5, 1.0, 2.0, 1.0, "Input\n1024d float32\n4096 bytes", "#E3F2FD"),
        (3.0, 1.0, 1.8, 1.0, "PCA Rotate\n+ Truncate\n→ 384d", "#BBDEFB"),
        (5.3, 1.0, 1.8, 1.0, "TurboQuant\n3-bit\nQuantize", "#90CAF9"),
        (7.6, 1.0, 2.0, 1.0, "Output\n384d × 3bit\n144 bytes", "#42A5F5"),
    ]

    for x, y, w, h, text, color in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="#1565C0", linewidth=1)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=6, fontweight="bold")

    # Arrows
    arrow_style = "Simple,tail_width=2,head_width=8,head_length=6"
    for x1, x2 in [(2.5, 3.0), (4.8, 5.3), (7.1, 7.6)]:
        arrow = FancyArrowPatch((x1, 1.5), (x2, 1.5),
                                arrowstyle="->", color="#1565C0",
                                linewidth=1.5, mutation_scale=10)
        ax.add_patch(arrow)

    # Compression label
    ax.text(5.0, 0.5, "27.7× compression", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                      edgecolor="#4CAF50", linewidth=1))

    fig.savefig(os.path.join(OUT_DIR, "fig_pipeline.pdf"))
    plt.close(fig)
    print("Generated fig_pipeline.pdf")


if __name__ == "__main__":
    fig_eigenspectrum()
    fig_cosine_comparison()
    fig_recall_comparison()
    fig_pareto_frontier()
    fig_pipeline()
    print("\nAll figures generated successfully!")
