# Breadth re-run — Qwen2.5-7B, 7 LongBench tasks, under the sidecar guard

The clean replacement for the retracted 7-task aggregate (CHANGELOG Errata
2026-08-15; paper `kv_tmlr` Table 3 caption). One GPU, three arms, resumable
per (arm, task); every cell writes a `config.0.json` sidecar and
`score_breadth.py` refuses to score a cell whose sidecar disagrees with its arm.

| file | role |
|---|---|
| `breadth_arm.py` | runs ONE (arm, task) cell through the harness's own `main()` — the 2026-08-08 re-validation pattern (`reval_arm.py`), plus a batch-probe `ThermalController` and the Volta SDPA accommodation |
| `breadth_run.sh` | loops arms × tasks on `CUDA_VISIBLE_DEVICES=${GPU:-1}`; never starts a cell above 75 °C; a watchdog SIGSTOPs the worker above 83 °C and resumes below 72 °C |
| `score_breadth.py` | LongBench's own `metrics.py`; verifies sidecars; writes `results_breadth_<SUFFIX>.json` |

Arms are exactly the `tq_expand.sh` env lines: `fp16` (`NOQUANT=1`), `nf4a` and
`nf4` (`KEY_BITS=4 VAL_BITS=4 GROUP=32 HOT=128 SINK=4 OUTLIER_FRAC=0.02 PREROPE=0`).
Data: `/archive/longbench/data` on Atlas (multifieldqa_en has 150 docs, the rest 200).

```bash
# on Atlas, from /archive/c12/breadth (harness = benchmarks/kvquant_matrix/tq_paper_lb_shard.py at HEAD)
SUFFIX=smoke NUM_SHARDS=100 TASKS=2wikimqa ARMS="fp16 nf4a nf4" bash breadth_run.sh   # 2 docs, ~90 s
tmux new -d -s breadth 'SUFFIX=full bash breadth_run.sh'                               # ~10-14 h
tmux capture-pane -p -t breadth | tail; cat breadth_full.log
```

Smoke 2026-09-03 (harness sha256 `605773ad9bc11a54`, transformers 5.5.0, torch
2.10.0+cu128): fp16 and nf4a return identical correct answers on docs 0/100 of
2wikimqa; symmetric nf4 returns wrong ones — the collapse acting as the positive
control that quantization is applied. Full run launched 2026-09-03 15:10 UTC in
tmux session `breadth`, GPU 1 (GPU 0 was holding a llama-server).
