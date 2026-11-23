"""The fitted models behind the registered predictions of Section sec:transfer.

    python paper/pvldb1t/figdata/transfer_fits.py benchmarks/fleet/record/1t/post \
        > paper/pvldb1t/figdata/transfer_fits.txt

Reads the two committed prediction records and nothing else:

- scale_transfer_predict_1tnm.json (benchmarks/fleet/scale_transfer.py): the
  calibration curve of nested subsets of 1, 2, 3, 5, 10 and 20 servers of the
  trillion-row index, each server 2e9 rows.
- scale_transfer_small_predict_1tnm.json (benchmarks/fleet/scale_transfer_small.py): the
  calibration curve of separately built corpora of 1e8, 2e8, 5e8 and 1e9 rows.

Both scripts fit by least squares in the natural logarithm of the row count N:

  M0  R(N*) = R(N_c), the value at the largest calibration size N_c
  M1  log(1 - R) = a + b log N           (a power law in the deficit)
  M2  R = min(1, a2 + b2 log N)          (linear in log N, capped at 1)

and predict at N* = 1e12. This script refits each model the same way, checks that it
gives the recorded prediction, and prints the fit in the form the paper uses, relative
to N_c:

  M1  1 - R(N) = D_c (N / N_c)^b,          D_c = exp(a + b log N_c)
  M2  R(N)     = min(1, R_c + g log10(N / N_c)),  R_c = a2 + b2 log N_c, g = b2 log 10
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

N_STAR = 1e12
ROWS_PER_SERVER = 2e9


def fit(n: np.ndarray, r: np.ndarray) -> dict:
    x = np.log(n)
    nc = float(n[-1])
    out = {"M0": float(r[-1])}
    b, a = np.polyfit(x, np.log(np.clip(1.0 - r, 1e-6, None)), 1)
    out["M1"] = float(1.0 - np.exp(a + b * math.log(N_STAR)))
    out["M1_exponent"] = float(b)
    out["M1_deficit_at_Nc"] = float(np.exp(a + b * math.log(nc)))
    b2, a2 = np.polyfit(x, r, 1)
    out["M2_uncapped"] = float(a2 + b2 * math.log(N_STAR))
    out["M2"] = min(1.0, out["M2_uncapped"])
    out["M2_per_decade"] = float(b2 * math.log(10.0))
    out["M2_value_at_Nc"] = float(a2 + b2 * math.log(nc))
    return out


def report(
    name: str, n: np.ndarray, curves: dict, recorded: dict, registered: dict
) -> None:
    print(f"== {name}")
    print("calibration sizes (rows): " + ", ".join(f"{v:.0e}" for v in n))
    for w in sorted(curves, key=int):
        r = np.array(curves[w], float)
        f = fit(n, r)
        for m in ("M0", "M1", "M2"):
            rec = recorded[w][m]
            assert abs(f[m] - rec) < 1e-9, (name, w, m, f[m], rec)
        reg = registered.get(w, "-")
        print(
            f"width {w:>3}: curve "
            + " ".join(f"{v:.4f}" for v in r)
            + f" | registered {reg}"
        )
        print(
            f"   M0 {f['M0']:.4f}"
            f" | M1 1-R = {f['M1_deficit_at_Nc']:.4f} (N/N_c)^{f['M1_exponent']:+.4f}"
            f" -> {f['M1']:.4f}"
            f" | M2 R = {f['M2_value_at_Nc']:.4f} {f['M2_per_decade']:+.4f}"
            " log10(N/N_c)"
            f" -> {f['M2_uncapped']:.4f}, capped {f['M2']:.4f}"
        )


def main() -> int:
    post = sys.argv[1]
    big = json.load(open(os.path.join(post, "scale_transfer_predict_1tnm.json")))
    w0 = next(iter(big["widths"]))
    ks = sorted(int(k) for k in big["widths"][w0]["calibration_curve"])
    n = np.array(ks, float) * ROWS_PER_SERVER
    curves = {
        w: [v["calibration_curve"][str(k)] for k in ks]
        for w, v in big["widths"].items()
    }
    recorded = {w: v["prediction"] for w, v in big["widths"].items()}
    print(f"record {post}")
    print(
        f"nested subsets of the index: servers {ks}, calibration servers "
        f"{big['calibration_servers']}"
    )
    report("registration 1 (in index)", n, curves, recorded, big["registered_models"])

    small = json.load(
        open(os.path.join(post, "scale_transfer_small_predict_1tnm.json"))
    )
    n = np.array(small["sizes_rows"], float)
    curves = {
        w: [v["curve"][str(int(s))] for s in small["sizes_rows"]]
        for w, v in small["widths"].items()
    }
    recorded = {w: v["prediction_1e12"] for w, v in small["widths"].items()}
    report(
        "registration 2 (separate corpora)",
        n,
        curves,
        recorded,
        small["registered_models"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
