# kvpress 0.5.5, four files, unchanged

Expected Attention (Devoto, Jeblick and Jégou) for Part V's `ea` and `ea_u3` baselines
(docs/PREREG_full_consumer_kv.md, section 3), from NVIDIA's kvpress under Apache-2.0
(`LICENSE`, copied from the wheel's `kvpress-0.5.5.dist-info/licenses/LICENSE`).

Source: `kvpress-0.5.5-py3-none-any.whl` from PyPI, sha256
`900bdb175b546f19c844a8d40e9e7bc6bff7f302689821263d376e31ff62f07e` (downloaded on Atlas,
2026-10-14). The files below are byte-identical to the wheel's; their sha256 values match
its RECORD and are pinned in `tests/test_part5_ea.py`.

| file | sha256 |
|---|---|
| `kvpress/utils.py` | `e8a19f23bfec0e06c058a9563b86f823935849ac7b93287657d602e4f694fce2` |
| `kvpress/presses/base_press.py` | `b37e5d08bce40ef90cd3dde1e9853ed7fa3fdc9996d42942a2dc6da0fd16fe1a` |
| `kvpress/presses/scorer_press.py` | `39191c9fb161238a2c44b32df8852e7406c3026423cd198470a0be026b8dfc50` |
| `kvpress/presses/expected_attention_press.py` | `6fbae9ab7b976e295959a53b80d145fac3687138cd91fa8fcf0e39d48f6f6c00` |

Why four files and not the package:

- kvpress 0.5.5 requires `transformers<5.3`; Part V's environment is Part II's
  (transformers 5.5.0), so the package is not installed.
- Its `__init__` imports every press, some of which need packages the environment does
  not have (`fire`, `peft`). `ea.py` loads these four modules under stub `kvpress`
  packages, so their own imports resolve to these files and nothing else runs.
- `BasePress.forward_hook` reads `kwargs["cache_position"]`, which transformers 5.5 no
  longer passes to the attention layer. `ea.py` replaces only that hook, with the same
  steps (prefill only; the layer's cached keys and values through
  `ExpectedAttentionPress.compress`; the result written back to the cache). Scoring and
  selection are kvpress's own code.
