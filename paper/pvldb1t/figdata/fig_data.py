"""Figure data for the PVLDB 1T paper, derived from the committed record files.

Reads, under benchmarks/fleet/record/1t/post/, score_1T.log (RESULT_JSON with 500 per-server
wall times for the reference scan and the routed passes at 32 and 128 probes), probe_1T.log
(the same for 16, 64 and 256 probes), analysis1t_cells.log (the cell census and the recall
predicted from cell ranks) and analysis1t_partials.log (per-query recall and where the
reference neighbours live), and writes next to itself

  ecdf_<phase>.dat   (seconds, share of servers at or below), 61 quantile points a phase
  reach.dat          width, predicted recall, measured recall, queries at recall 1
  stats.json         every statistic the prose quotes

Nothing in the paper's tables is typed in by hand where this script can derive it.
"""

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "benchmarks", "fleet", "record", "1t", "post")
)


def tagged_json(path, tag):
    txt = open(os.path.join(REC, path), encoding="utf-8").read()
    m = re.search(tag + r" (\{.*\})", txt)
    return json.loads(m.group(1))


main = tagged_json("score_1T.log", "RESULT_JSON")
probe = tagged_json("probe_1T.log", "RESULT_JSON")
cells = tagged_json("analysis1t_cells.log", "CELLS_JSON")
partials = tagged_json("analysis1t_partials.log", "ANALYSIS_JSON")

phases = {
    "exact scan": np.array(main["reference_wall_s_per_server"], float),
    "routed 16": np.array(probe["ivf"]["16"]["wall_s_per_server"], float),
    "routed 32": np.array(main["ivf"]["32"]["wall_s_per_server"], float),
    "routed 64": np.array(probe["ivf"]["64"]["wall_s_per_server"], float),
    "routed 128": np.array(main["ivf"]["128"]["wall_s_per_server"], float),
    "routed 256": np.array(probe["ivf"]["256"]["wall_s_per_server"], float),
}
recall = {
    "16": probe["ivf"]["16"]["recall_vs_adc_fullscan"],
    "32": main["ivf"]["32"]["recall_vs_adc_fullscan"],
    "64": probe["ivf"]["64"]["recall_vs_adc_fullscan"],
    "128": main["ivf"]["128"]["recall_vs_adc_fullscan"],
    "256": probe["ivf"]["256"]["recall_vs_adc_fullscan"],
}

stats = {"recall_vs_exact_scan": {k: round(v, 4) for k, v in recall.items()}, "phases": {}}
for name, w in phases.items():
    assert len(w) == 500, (name, len(w))
    stats["phases"][name] = {
        "n": int(len(w)),
        "median_s": round(float(np.median(w))),
        "mean_s": round(float(w.mean())),
        "p10_s": round(float(np.percentile(w, 10))),
        "p90_s": round(float(np.percentile(w, 90))),
        "min_s": round(float(w.min())),
        "max_s": round(float(w.max())),
        "cpu_hours": round(float(w.sum() / 3600)),
        "servers_over_2x_median": int((w > 2 * np.median(w)).sum()),
    }
    fn = "ecdf_" + name.replace(" ", "_") + ".dat"
    with open(os.path.join(HERE, fn), "w", encoding="utf-8") as f:
        s = np.sort(w)
        for q in np.linspace(0, 1, 61):
            f.write(f"{float(np.quantile(s, q)):.0f} {q:.4f}\n")

stats["total_cpu_hours_main_run"] = round(
    float(sum(phases[k].sum() for k in ("exact scan", "routed 32", "routed 128")) / 3600)
)
stats["total_cpu_hours_probe_sweep"] = round(
    float(sum(phases[k].sum() for k in ("routed 16", "routed 64", "routed 256")) / 3600)
)
stats["total_cpu_hours_all"] = round(float(sum(w.sum() for w in phases.values()) / 3600))

pred = cells["reachability"]["predicted_recall_at_probe_width"]
q1 = cells["reachability"]["queries_at_1_at_width"]
stats["reachability"] = {
    "predicted": pred,
    "queries_at_1": q1,
    "cell_rank_of_reference_neighbours": cells["reachability"][
        "cell_rank_of_reference_neighbours"
    ],
    "measured_vs_predicted": cells["measured_vs_predicted"],
}
stats["cell_sizes"] = cells["cell_sizes"]
stats["census_wall_s"] = cells["per_server_wall_s"]
stats["per_query_recall"] = partials["per_query_recall"]
stats["server_drop"] = partials["server_drop"]
with open(os.path.join(HERE, "reach.dat"), "w", encoding="utf-8") as f:
    f.write("width predicted measured queries\n")
    for wdt, v in pred.items():
        f.write(f"{wdt} {v} {recall.get(wdt, 'nan')} {q1[wdt]}\n")

with open(os.path.join(HERE, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=1))
