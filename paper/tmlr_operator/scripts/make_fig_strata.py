"""Figure 1: how each of the four consuming operators treats an error. A diagram.

Run from the repository root:
    python paper/tmlr_operator/scripts/make_fig_strata.py
Writes paper/tmlr_operator/fig_strata.pdf and .png. No run record is read. The numbers in
panel (a) are softmax values of the three example score rows, computed here; panel (d) uses
an illustrative range of decay rates. Nothing in the figure is a measurement.

  (a) softmax        a constant added to a row of scores changes nothing; a centered change does
  (b) linear write   value and output add to the residual stream, so every error direction counts
  (c) top-k          three logits in the plane orthogonal to the common-mode direction; the
                     selection changes only when a wall is crossed
  (d) recurrence     the gain 1/(1-a) diverges at a = 1; where 8 grid levels fall under two bases
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "mathtext.fontset": "cm"})
BLUE, ORANGE, GREY = "#4477aa", "#ee9944", "0.55"
WHITE = dict(fc="white", ec="none", pad=1.0, alpha=0.92)

fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.7))
(ax_a, ax_b), (ax_c, ax_d) = axes


def softmax(v):
    e = np.exp(np.asarray(v) - max(v))
    return e / e.sum()


# ---------------------------------------------------------------- (a) softmax
ax = ax_a
ax.set_title("(a) softmax over a row of scores", loc="left", fontsize=9.5)
rows = [("scores", [2.0, 1.0, 0.5], BLUE, ""),
        ("+1 on every score", [3.0, 2.0, 1.5], BLUE, ""),
        ("+0.8 on one score", [2.8, 1.0, 0.5], ORANGE, "//")]
x0 = [0.0, 1.55, 3.1]
for (label, sc, col, hatch), xo in zip(rows, x0):
    p = softmax(sc)
    for k, v in enumerate(p):
        ax.add_patch(Rectangle((xo + 0.32 * k, 0), 0.26, v, facecolor=col, edgecolor="black", lw=0.6, hatch=hatch))
        ax.text(xo + 0.32 * k + 0.13, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.text(xo + 0.45, -0.07, label, ha="center", va="top", fontsize=8)
    ax.text(xo + 0.45, -0.17, "[" + ", ".join(f"{s:g}" for s in sc) + "]", ha="center", va="top", fontsize=8)
ax.annotate("", (1.5, 0.83), (1.0, 0.83), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("", (3.05, 0.83), (2.55, 0.83), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(2.0, 0.95, "row constant:\nunchanged", ha="center", va="center", fontsize=7.5)
ax.text(3.6, 0.99, "centered:\nweights move", ha="center", va="center", fontsize=7.5)
ax.text(0.45, 0.94, "attention\nweights", ha="center", va="center", fontsize=8)
ax.set_xlim(-0.15, 4.2)
ax.set_ylim(-0.3, 1.05)
ax.axis("off")

# ---------------------------------------------------------------- (b) linear write
ax = ax_b
ax.set_title("(b) linear write to the residual stream", loc="left", fontsize=9.5)
ax.add_patch(Rectangle((0.05, 0.47), 0.9, 0.06, facecolor=GREY, edgecolor="none"))
ax.text(0.5, 0.40, "residual stream", ha="center", va="top", fontsize=8.5)
for xc, name, col, hatch in [(0.27, "value $W^V$", BLUE, ""), (0.73, "output $W^O$", ORANGE, "//")]:
    ax.add_patch(Rectangle((xc - 0.13, 0.80), 0.26, 0.13, facecolor="white", edgecolor="black", lw=0.8))
    ax.text(xc, 0.865, name, ha="center", va="center", fontsize=8.5)
    ax.add_patch(FancyArrowPatch((xc, 0.80), (xc, 0.535), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="black"))
    ax.add_patch(Rectangle((xc - 0.045, 0.60), 0.09, 0.12, facecolor=col, edgecolor="black", lw=0.6, hatch=hatch))
ax.text(0.5, 0.66, "error\nadds", ha="center", va="center", fontsize=8)
ax.text(0.5, 0.20, "No softmax stands between the error and the stream.\nNo direction of the error is cancelled.",
        ha="center", va="center", fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0.05, 1.0)
ax.axis("off")

# ---------------------------------------------------------------- (c) top-k selection
ax = ax_c
ax.set_title("(c) top-$k$ selection: three logits", loc="left", fontsize=9.5)
R = 1.0


def ray(deg, r=R):
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)


regions = [(-30, 90, "0.94", "", "logit 1\nlargest", (-2, 0.62)),
           (90, 210, "0.84", "--", "logit 2\nlargest", (176, 0.62)),
           (210, 330, "0.74", "//", "logit 3\nlargest", (270, 0.62))]
for a0, a1, shade, hatch, label, (ld, lr) in regions:
    ax.add_patch(Wedge((0, 0), R, a0, a1, facecolor=shade, edgecolor="0.6", hatch=hatch, lw=0))
    x, y = ray(ld, lr)
    ax.text(x, y, label, ha="center", va="center", fontsize=8, bbox=WHITE)
for deg, lab in [(90, r"$\ell_1=\ell_2$"), (330, r"$\ell_1=\ell_3$"), (210, r"$\ell_2=\ell_3$")]:
    x, y = ray(deg)
    ax.plot([0, x], [0, y], color="black", lw=2.0, solid_capstyle="butt")
    tx, ty = ray(deg, 1.13)
    ax.text(tx, ty, lab, ha="center", va="center", fontsize=9)
for deg in (270, 150, 30):
    x, y = ray(deg)
    ax.plot([0, x], [0, y], color="black", lw=0.9, ls=(0, (5, 3)))
ax.plot([0], [0], marker="o", ms=4, color="black")
RADIUS = 0.13
# square: nearest top-1 wall is the vertical one, at distance 0.29 > RADIUS
p = (0.62 * math.cos(math.radians(62)), 0.62 * math.sin(math.radians(62)))
ax.add_patch(Circle(p, RADIUS, fill=False, lw=0.9, color="black"))
ax.plot(*p, marker="s", ms=4.5, color="black")
ax.annotate("", (0.0, p[1]), p, arrowprops=dict(arrowstyle="|-|,widthA=0.22,widthB=0.22", lw=0.9, shrinkA=0, shrinkB=0))
ax.text(p[0] / 2 + 0.03, p[1] - 0.045, "margin", ha="center", va="top", fontsize=7.5, bbox=WHITE)
ax.text(p[0] + RADIUS + 0.03, p[1], "kept", ha="left", va="center", fontsize=8, bbox=WHITE)
# triangle: distance 0.06 < RADIUS from the wall between logits 2 and 1
q = (-0.06, 0.28)
ax.add_patch(Circle(q, RADIUS, fill=False, lw=0.9, color="black"))
ax.plot(*q, marker="^", ms=5.5, color="black")
ax.annotate("can change", q, (-0.92, 0.36), fontsize=8, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5), bbox=WHITE)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.2, 1.25)
ax.set_aspect("equal")
ax.axis("off")

# ---------------------------------------------------------------- (d) recurrence
ax = ax_d
ax.set_title("(d) linear recurrence $h_t = a\\,h_{t-1} + b_t$", loc="left", fontsize=9.5)
a = np.linspace(0.0, 0.995, 600)
ax.plot(a, 1 / (1 - a), color="black", lw=1.4)
ax.set_yscale("log")
ax.set_ylim(0.07, 400)
ax.set_xlim(-0.02, 1.02)
ax.set_xlabel("decay $a$ per step")
ax.set_ylabel("steady-state gain $1/(1-a)$")
ax.axvline(1.0, color="black", lw=0.6, ls=":")
# where 8 levels fall, for decay rates lambda = -ln a between 1e-3 and 10 (illustrative)
lam_lin = np.linspace(1e-3, 10.0, 8)
lam_log = np.exp(np.linspace(math.log(1e-3), math.log(10.0), 8))
ax.plot(np.exp(-lam_lin), np.full(8, 0.32), ls="none", marker="|", ms=11, mew=1.6, color=ORANGE)
ax.plot(np.exp(-lam_log), np.full(8, 0.13), ls="none", marker="|", ms=11, mew=1.6, color=BLUE)
ax.text(0.62, 0.32, "8 levels uniform in the rate $-\\ln a$", ha="center", va="center", fontsize=7.5)
ax.text(0.275, 0.13, "8 levels uniform\nin $\\ln(-\\ln a)$", ha="center", va="center", fontsize=7.5)
ax.annotate("gain diverges\nat $a=1$", (0.985, 66), (0.42, 90), fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.7))
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.tight_layout(h_pad=1.6, w_pad=1.4)
fig.savefig("paper/tmlr_operator/fig_strata.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("paper/tmlr_operator/fig_strata.png", dpi=170, bbox_inches="tight", pad_inches=0.02)
print("wrote paper/tmlr_operator/fig_strata.pdf")
print("softmax rows:", [np.round(softmax(r[1]), 3).tolist() for r in rows])
print("levels linear in rate, a:", np.round(np.exp(-lam_lin), 3).tolist())
print("levels log in rate, a:", np.round(np.exp(-lam_log), 3).tolist())
