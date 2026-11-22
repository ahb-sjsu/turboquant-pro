"""Figure: the stratification of logit space by ties, for three experts.

Run from the repository root:
    python paper/tmlr_operator/scripts/make_fig_strata.py
Writes paper/tmlr_operator/fig_strata.pdf. No data is read; the figure is a diagram.

Three logits (l1, l2, l3) are drawn in the plane orthogonal to (1, 1, 1), the common-mode
direction, which no selection depends on. The three lines l_i = l_j meet at the point where
all three tie. Its faces are that point, six open rays and six open sectors.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})

fig, ax = plt.subplots(figsize=(5.2, 4.4))
R = 1.0


def ray(deg, r=R):
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)


# The three regions where one expert has the largest logit. Each is two sectors and the ray
# between them. Hatching, not colour alone, tells them apart.
regions = [(-30, 90, "0.93", "", "expert 1 largest", (3, 0.62)),
           (90, 210, "0.84", "..", "expert 2 largest", (142, 0.62)),
           (210, 330, "0.75", "//", "expert 3 largest", (270, 0.66))]
for a0, a1, shade, hatch, label, (lab_deg, lab_r) in regions:
    ax.add_patch(Wedge((0, 0), R, a0, a1, facecolor=shade, edgecolor="0.55", hatch=hatch, lw=0))
    x, y = ray(lab_deg, lab_r)
    ax.text(x, y, label, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))

# Walls of the top-1 selection: the two largest logits tie.
for deg, lab in [(90, r"$\ell_1=\ell_2>\ell_3$"), (330, r"$\ell_1=\ell_3>\ell_2$"), (210, r"$\ell_2=\ell_3>\ell_1$")]:
    x, y = ray(deg)
    ax.plot([0, x], [0, y], color="black", lw=2.0, solid_capstyle="butt")
    tx, ty = ray(deg, 1.13)
    ax.text(tx, ty, lab, ha="center", va="center")
# The other three rays: the two smallest logits tie. Faces of the arrangement, and the walls
# of the top-2 selection, but the top-1 selection does not change across them.
for deg, lab in [(270, r"$\ell_3>\ell_1=\ell_2$"), (150, r"$\ell_2>\ell_1=\ell_3$"), (30, r"$\ell_1>\ell_2=\ell_3$")]:
    x, y = ray(deg)
    ax.plot([0, x], [0, y], color="black", lw=1.0, ls=(0, (4, 3)))
    tx, ty = ray(deg, 1.13)
    ax.text(tx, ty, lab, ha="center", va="center", color="0.25")
ax.plot([0], [0], marker="o", ms=5, color="black")
ax.text(0.06, -0.06, "all three tie", ha="left", va="top", fontsize=8, bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))

# A logit vector well inside a region: its margin is the distance to the nearest wall.
# At 50 degrees and radius 0.5 the nearest wall is the vertical one, at distance 0.32.
p = (0.5 * math.cos(math.radians(50)), 0.5 * math.sin(math.radians(50)))
ax.plot(*p, marker="s", ms=5, color="black")
ax.add_patch(Circle(p, 0.13, fill=False, lw=0.9, color="black"))
ax.plot([p[0], 0.0], [p[1], p[1]], color="black", lw=0.9, ls=":")
ax.text(p[0] / 2, p[1] - 0.045, "margin", ha="center", va="top", fontsize=8,
        bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
ax.text(p[0] + 0.17, p[1] - 0.01, r"$R<1$: route kept", ha="left", va="center", fontsize=8,
        bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
# A logit vector near a wall: the same perturbation reaches across it.
q = (-0.06, 0.55)
ax.plot(*q, marker="^", ms=6, color="black")
ax.add_patch(Circle(q, 0.13, fill=False, lw=0.9, color="black"))
ax.annotate(r"$R>1$: route can flip", q, (-0.86, 0.80), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5),
            bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))

ax.plot([], [], color="black", lw=2.0, label="wall of the top-1 selection")
ax.plot([], [], color="black", lw=1.0, ls=(0, (4, 3)), label="wall of the top-2 selection")
ax.legend(loc="lower left", bbox_to_anchor=(-0.16, -0.14), frameon=False, fontsize=8, handlelength=2.6)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.28, 1.28)
ax.set_aspect("equal")
ax.axis("off")
fig.savefig("paper/tmlr_operator/fig_strata.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("paper/tmlr_operator/fig_strata.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote paper/tmlr_operator/fig_strata.pdf")
