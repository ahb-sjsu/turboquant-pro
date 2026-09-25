## Gates

- **G0**: PASS
  - msmarco P k=64: 0.3022 vs 0.3022
  - msmarco P k=128: 0.5635 vs 0.5635
  - msmarco P k=256: 0.7643 vs 0.7643
  - msmarco O k=64: 0.3169 vs 0.3169
  - msmarco O k=128: 0.5760 vs 0.5760
  - msmarco O k=256: 0.7775 vs 0.7775
  - hotpotqa P k=64: 0.1775 vs 0.1775
  - hotpotqa P k=128: 0.4083 vs 0.4083
  - hotpotqa P k=256: 0.6715 vs 0.6715
  - hotpotqa O k=64: 0.2230 vs 0.2230
  - hotpotqa O k=128: 0.4385 vs 0.4385
  - hotpotqa O k=256: 0.6971 vs 0.6970
  - split identity fiqa k=128: O - Oey = +0.0000
  - split identity fiqa k=256: O - Oey = +0.0000
  - split identity fiqa k=64: O - Oey = +0.0000
  - split identity hotpotqa k=128: O - Oey = +0.0000
  - split identity hotpotqa k=256: O - Oey = +0.0000
  - split identity hotpotqa k=64: O - Oey = +0.0000
  - split identity hotpotqa-sym k=128: O - Oey = -0.0000
  - split identity hotpotqa-sym k=256: O - Oey = +0.0000
  - split identity hotpotqa-sym k=64: O - Oey = +0.0000
  - split identity msmarco k=128: O - Oey = +0.0000
  - split identity msmarco k=256: O - Oey = +0.0000
  - split identity msmarco k=64: O - Oey = +0.0000
  - split identity msmarco-sym k=128: O - Oey = +0.0000
  - split identity msmarco-sym k=256: O - Oey = -0.0000
  - split identity msmarco-sym k=64: O - Oey = +0.0000
  - split identity nq k=128: O - Oey = +0.0000
  - split identity nq k=256: O - Oey = +0.0000
  - split identity nq k=64: O - Oey = +0.0000
  - split identity quora k=128: O - Oey = +0.0000
  - split identity quora k=256: O - Oey = +0.0000
  - split identity quora k=64: O - Oey = +0.0000
- **G1**: PASS
  - BETTER in 6 of 6
- **G2**: PASS
- **G3**: PASS

## Registered verdicts (single-pass recall@10)

- **TQ H1**: HOLDS {'BETTER': 12}
  - msmarco O-P k=64 b=2: +0.0143 [+0.0120, +0.0164] BETTER
  - msmarco O-P k=64 b=4: +0.0157 [+0.0133, +0.0180] BETTER
  - msmarco O-P k=128 b=2: +0.0125 [+0.0102, +0.0147] BETTER
  - msmarco O-P k=128 b=4: +0.0123 [+0.0104, +0.0142] BETTER
  - msmarco O-P k=256 b=2: +0.0138 [+0.0117, +0.0158] BETTER
  - msmarco O-P k=256 b=4: +0.0125 [+0.0109, +0.0142] BETTER
  - hotpotqa O-P k=64 b=2: +0.0328 [+0.0309, +0.0348] BETTER
  - hotpotqa O-P k=64 b=4: +0.0420 [+0.0396, +0.0445] BETTER
  - hotpotqa O-P k=128 b=2: +0.0245 [+0.0224, +0.0266] BETTER
  - hotpotqa O-P k=128 b=4: +0.0294 [+0.0271, +0.0316] BETTER
  - hotpotqa O-P k=256 b=2: +0.0161 [+0.0141, +0.0183] BETTER
  - hotpotqa O-P k=256 b=4: +0.0217 [+0.0198, +0.0236] BETTER
- **TQ H2**: HOLDS {'TIE': 6}
  - msmarco-sym O-P k=64 b=2: -0.0004 [-0.0025, +0.0017] TIE
  - msmarco-sym O-P k=64 b=4: -0.0029 [-0.0047, -0.0011] TIE
  - msmarco-sym O-P k=128 b=2: -0.0009 [-0.0030, +0.0013] TIE
  - msmarco-sym O-P k=128 b=4: +0.0004 [-0.0010, +0.0019] TIE
  - msmarco-sym O-P k=256 b=2: -0.0010 [-0.0029, +0.0010] TIE
  - msmarco-sym O-P k=256 b=4: -0.0000 [-0.0014, +0.0013] TIE
- **TQ H3**: MIXED rho=+0.214
- **TQ H4**: HOLDS {'BETTER': 6}
  - msmarco O-Of k=64 b=2: +0.0536 [+0.0504, +0.0568] BETTER
  - msmarco O-Of k=64 b=4: +0.0602 [+0.0563, +0.0642] BETTER
  - msmarco O-Of k=128 b=2: +0.0502 [+0.0472, +0.0532] BETTER
  - msmarco O-Of k=128 b=4: +0.0410 [+0.0377, +0.0443] BETTER
  - msmarco O-Of k=256 b=2: +0.0389 [+0.0362, +0.0416] BETTER
  - msmarco O-Of k=256 b=4: +0.0227 [+0.0200, +0.0254] BETTER
- **RBQ H1**: REFUTED {'BETTER': 8, 'TIE': 2, 'WORSE': 2}
  - msmarco O-P k=64 b=2: +0.0279 [+0.0247, +0.0311] BETTER
  - msmarco O-P k=64 b=4: +0.0158 [+0.0130, +0.0185] BETTER
  - msmarco O-P k=128 b=2: +0.0273 [+0.0240, +0.0306] BETTER
  - msmarco O-P k=128 b=4: +0.0136 [+0.0112, +0.0161] BETTER
  - msmarco O-P k=256 b=2: +0.0348 [+0.0315, +0.0380] BETTER
  - msmarco O-P k=256 b=4: +0.0183 [+0.0160, +0.0207] BETTER
  - hotpotqa O-P k=64 b=2: -0.0005 [-0.0034, +0.0025] TIE
  - hotpotqa O-P k=64 b=4: +0.0323 [+0.0295, +0.0352] BETTER
  - hotpotqa O-P k=128 b=2: -0.0245 [-0.0278, -0.0211] WORSE
  - hotpotqa O-P k=128 b=4: +0.0160 [+0.0133, +0.0188] BETTER
  - hotpotqa O-P k=256 b=2: -0.0161 [-0.0197, -0.0125] WORSE
  - hotpotqa O-P k=256 b=4: +0.0043 [+0.0015, +0.0069] TIE
- **RBQ H2**: HOLDS {'TIE': 6}
  - msmarco-sym O-P k=64 b=2: -0.0024 [-0.0053, +0.0004] TIE
  - msmarco-sym O-P k=64 b=4: -0.0027 [-0.0047, -0.0006] TIE
  - msmarco-sym O-P k=128 b=2: -0.0013 [-0.0042, +0.0016] TIE
  - msmarco-sym O-P k=128 b=4: -0.0011 [-0.0031, +0.0008] TIE
  - msmarco-sym O-P k=256 b=2: -0.0026 [-0.0053, +0.0002] TIE
  - msmarco-sym O-P k=256 b=4: -0.0002 [-0.0021, +0.0017] TIE
- **RBQ H3**: MIXED rho=+0.071
- **RBQ H4**: HOLDS {'BETTER': 6}
  - msmarco O-Of k=64 b=2: +0.1273 [+0.1229, +0.1318] BETTER
  - msmarco O-Of k=64 b=4: +0.0796 [+0.0753, +0.0840] BETTER
  - msmarco O-Of k=128 b=2: +0.1016 [+0.0974, +0.1058] BETTER
  - msmarco O-Of k=128 b=4: +0.0534 [+0.0496, +0.0572] BETTER
  - msmarco O-Of k=256 b=2: +0.0734 [+0.0695, +0.0773] BETTER
  - msmarco O-Of k=256 b=4: +0.0381 [+0.0349, +0.0414] BETTER
- **OPQ H1**: HOLDS {'BETTER': 12}
  - msmarco O-P k=64 b=2: +0.0145 [+0.0120, +0.0169] BETTER
  - msmarco O-P k=64 b=4: +0.0141 [+0.0117, +0.0164] BETTER
  - msmarco O-P k=128 b=2: +0.0136 [+0.0112, +0.0160] BETTER
  - msmarco O-P k=128 b=4: +0.0139 [+0.0120, +0.0158] BETTER
  - msmarco O-P k=256 b=2: +0.0109 [+0.0088, +0.0130] BETTER
  - msmarco O-P k=256 b=4: +0.0133 [+0.0116, +0.0150] BETTER
  - hotpotqa O-P k=64 b=2: +0.0248 [+0.0227, +0.0269] BETTER
  - hotpotqa O-P k=64 b=4: +0.0405 [+0.0379, +0.0430] BETTER
  - hotpotqa O-P k=128 b=2: +0.0197 [+0.0174, +0.0219] BETTER
  - hotpotqa O-P k=128 b=4: +0.0276 [+0.0254, +0.0299] BETTER
  - hotpotqa O-P k=256 b=2: +0.0158 [+0.0136, +0.0180] BETTER
  - hotpotqa O-P k=256 b=4: +0.0207 [+0.0188, +0.0227] BETTER
- **OPQ H2**: HOLDS {'TIE': 6}
  - msmarco-sym O-P k=64 b=2: -0.0013 [-0.0035, +0.0010] TIE
  - msmarco-sym O-P k=64 b=4: -0.0028 [-0.0046, -0.0009] TIE
  - msmarco-sym O-P k=128 b=2: -0.0025 [-0.0047, -0.0002] TIE
  - msmarco-sym O-P k=128 b=4: -0.0000 [-0.0015, +0.0015] TIE
  - msmarco-sym O-P k=256 b=2: -0.0005 [-0.0025, +0.0015] TIE
  - msmarco-sym O-P k=256 b=4: +0.0001 [-0.0012, +0.0014] TIE
- **OPQ H3**: MIXED rho=+0.214
- **OPQ H4**: HOLDS {'BETTER': 6}
  - msmarco O-Of k=64 b=2: +0.0561 [+0.0528, +0.0594] BETTER
  - msmarco O-Of k=64 b=4: +0.0618 [+0.0578, +0.0658] BETTER
  - msmarco O-Of k=128 b=2: +0.0481 [+0.0450, +0.0512] BETTER
  - msmarco O-Of k=128 b=4: +0.0404 [+0.0370, +0.0438] BETTER
  - msmarco O-Of k=256 b=2: +0.0355 [+0.0328, +0.0382] BETTER
  - msmarco O-Of k=256 b=4: +0.0244 [+0.0217, +0.0272] BETTER

**Platform claim: HOLDS FOR TQ, OPQ ONLY**

## Reported, not scored: the same hypotheses after 5x rerank

- TQ: H1 HOLDS, H2 HOLDS, H3 MIXED, H4 HOLDS
- RBQ: H1 MIXED, H2 HOLDS, H3 MIXED, H4 HOLDS
- OPQ: H1 HOLDS, H2 HOLDS, H3 MIXED, H4 HOLDS

## Observer Advantage table (descriptive; measured byte levels only)

| family | arm | Q | endpoint | bytes P | bytes O | P/O |
|---|---|---:|---|---:|---:|---|
| TQ | msmarco | 0.80 | single | None | None | not measured |
| TQ | msmarco | 0.90 | single | None | None | not measured |
| TQ | msmarco | 0.95 | single | None | None | not measured |
| TQ | hotpotqa | 0.80 | single | None | None | not measured |
| TQ | hotpotqa | 0.90 | single | None | None | not measured |
| TQ | hotpotqa | 0.95 | single | None | None | not measured |
| TQ | msmarco-sym | 0.80 | single | None | None | not measured |
| TQ | msmarco-sym | 0.90 | single | None | None | not measured |
| TQ | msmarco-sym | 0.95 | single | None | None | not measured |
| TQ | hotpotqa-sym | 0.80 | single | None | None | not measured |
| TQ | hotpotqa-sym | 0.90 | single | None | None | not measured |
| TQ | hotpotqa-sym | 0.95 | single | None | None | not measured |
| TQ | fiqa | 0.80 | single | None | 132 | not measured |
| TQ | fiqa | 0.90 | single | None | None | not measured |
| TQ | fiqa | 0.95 | single | None | None | not measured |
| TQ | nq | 0.80 | single | None | None | not measured |
| TQ | nq | 0.90 | single | None | None | not measured |
| TQ | nq | 0.95 | single | None | None | not measured |
| TQ | quora | 0.80 | single | 132 | 132 | 1.00 |
| TQ | quora | 0.90 | single | None | None | not measured |
| TQ | quora | 0.95 | single | None | None | not measured |
| RBQ | msmarco | 0.80 | single | None | None | not measured |
| RBQ | msmarco | 0.90 | single | None | None | not measured |
| RBQ | msmarco | 0.95 | single | None | None | not measured |
| RBQ | hotpotqa | 0.80 | single | None | None | not measured |
| RBQ | hotpotqa | 0.90 | single | None | None | not measured |
| RBQ | hotpotqa | 0.95 | single | None | None | not measured |
| RBQ | msmarco-sym | 0.80 | single | None | None | not measured |
| RBQ | msmarco-sym | 0.90 | single | None | None | not measured |
| RBQ | msmarco-sym | 0.95 | single | None | None | not measured |
| RBQ | hotpotqa-sym | 0.80 | single | None | None | not measured |
| RBQ | hotpotqa-sym | 0.90 | single | None | None | not measured |
| RBQ | hotpotqa-sym | 0.95 | single | None | None | not measured |
| RBQ | fiqa | 0.80 | single | None | None | not measured |
| RBQ | fiqa | 0.90 | single | None | None | not measured |
| RBQ | fiqa | 0.95 | single | None | None | not measured |
| RBQ | nq | 0.80 | single | None | None | not measured |
| RBQ | nq | 0.90 | single | None | None | not measured |
| RBQ | nq | 0.95 | single | None | None | not measured |
| RBQ | quora | 0.80 | single | 148 | 148 | 1.00 |
| RBQ | quora | 0.90 | single | None | None | not measured |
| RBQ | quora | 0.95 | single | None | None | not measured |
| OPQ | msmarco | 0.80 | single | None | None | not measured |
| OPQ | msmarco | 0.90 | single | None | None | not measured |
| OPQ | msmarco | 0.95 | single | None | None | not measured |
| OPQ | hotpotqa | 0.80 | single | None | None | not measured |
| OPQ | hotpotqa | 0.90 | single | None | None | not measured |
| OPQ | hotpotqa | 0.95 | single | None | None | not measured |
| OPQ | msmarco-sym | 0.80 | single | None | None | not measured |
| OPQ | msmarco-sym | 0.90 | single | None | None | not measured |
| OPQ | msmarco-sym | 0.95 | single | None | None | not measured |
| OPQ | hotpotqa-sym | 0.80 | single | None | None | not measured |
| OPQ | hotpotqa-sym | 0.90 | single | None | None | not measured |
| OPQ | hotpotqa-sym | 0.95 | single | None | None | not measured |
| OPQ | fiqa | 0.80 | single | None | 128 | not measured |
| OPQ | fiqa | 0.90 | single | None | None | not measured |
| OPQ | fiqa | 0.95 | single | None | None | not measured |
| OPQ | nq | 0.80 | single | None | None | not measured |
| OPQ | nq | 0.90 | single | None | None | not measured |
| OPQ | nq | 0.95 | single | None | None | not measured |
| OPQ | quora | 0.80 | single | 128 | 128 | 1.00 |
| OPQ | quora | 0.90 | single | None | None | not measured |
| OPQ | quora | 0.95 | single | None | None | not measured |
| TQ | msmarco | 0.80 | rr5 | 68 | 68 | 1.00 |
| TQ | msmarco | 0.90 | rr5 | 68 | 68 | 1.00 |
| TQ | msmarco | 0.95 | rr5 | 132 | 132 | 1.00 |
| TQ | hotpotqa | 0.80 | rr5 | 68 | 68 | 1.00 |
| TQ | hotpotqa | 0.90 | rr5 | 132 | 132 | 1.00 |
| TQ | hotpotqa | 0.95 | rr5 | None | None | not measured |
| TQ | msmarco-sym | 0.80 | rr5 | 36 | 36 | 1.00 |
| TQ | msmarco-sym | 0.90 | rr5 | 68 | 68 | 1.00 |
| TQ | msmarco-sym | 0.95 | rr5 | 68 | 68 | 1.00 |
| TQ | hotpotqa-sym | 0.80 | rr5 | 68 | 68 | 1.00 |
| TQ | hotpotqa-sym | 0.90 | rr5 | 68 | 68 | 1.00 |
| TQ | hotpotqa-sym | 0.95 | rr5 | 132 | 132 | 1.00 |
| TQ | fiqa | 0.80 | rr5 | 36 | 36 | 1.00 |
| TQ | fiqa | 0.90 | rr5 | 68 | 68 | 1.00 |
| TQ | fiqa | 0.95 | rr5 | 68 | 68 | 1.00 |
| TQ | nq | 0.80 | rr5 | 68 | 68 | 1.00 |
| TQ | nq | 0.90 | rr5 | 68 | 68 | 1.00 |
| TQ | nq | 0.95 | rr5 | 132 | 132 | 1.00 |
| TQ | quora | 0.80 | rr5 | 36 | 36 | 1.00 |
| TQ | quora | 0.90 | rr5 | 68 | 68 | 1.00 |
| TQ | quora | 0.95 | rr5 | 68 | 68 | 1.00 |
| RBQ | msmarco | 0.80 | rr5 | 84 | 52 | 1.62 |
| RBQ | msmarco | 0.90 | rr5 | 84 | 84 | 1.00 |
| RBQ | msmarco | 0.95 | rr5 | 148 | 84 | 1.76 |
| RBQ | hotpotqa | 0.80 | rr5 | 84 | 84 | 1.00 |
| RBQ | hotpotqa | 0.90 | rr5 | 148 | 148 | 1.00 |
| RBQ | hotpotqa | 0.95 | rr5 | None | None | not measured |
| RBQ | msmarco-sym | 0.80 | rr5 | 52 | 52 | 1.00 |
| RBQ | msmarco-sym | 0.90 | rr5 | 84 | 84 | 1.00 |
| RBQ | msmarco-sym | 0.95 | rr5 | 84 | 84 | 1.00 |
| RBQ | hotpotqa-sym | 0.80 | rr5 | 84 | 84 | 1.00 |
| RBQ | hotpotqa-sym | 0.90 | rr5 | 84 | 84 | 1.00 |
| RBQ | hotpotqa-sym | 0.95 | rr5 | 148 | 148 | 1.00 |
| RBQ | fiqa | 0.80 | rr5 | 52 | 52 | 1.00 |
| RBQ | fiqa | 0.90 | rr5 | 84 | 84 | 1.00 |
| RBQ | fiqa | 0.95 | rr5 | 84 | 84 | 1.00 |
| RBQ | nq | 0.80 | rr5 | 52 | 52 | 1.00 |
| RBQ | nq | 0.90 | rr5 | 84 | 84 | 1.00 |
| RBQ | nq | 0.95 | rr5 | 84 | 84 | 1.00 |
| RBQ | quora | 0.80 | rr5 | 52 | 52 | 1.00 |
| RBQ | quora | 0.90 | rr5 | 52 | 52 | 1.00 |
| RBQ | quora | 0.95 | rr5 | 84 | 84 | 1.00 |
| OPQ | msmarco | 0.80 | rr5 | 64 | 64 | 1.00 |
| OPQ | msmarco | 0.90 | rr5 | 64 | 64 | 1.00 |
| OPQ | msmarco | 0.95 | rr5 | 128 | 128 | 1.00 |
| OPQ | hotpotqa | 0.80 | rr5 | 64 | 64 | 1.00 |
| OPQ | hotpotqa | 0.90 | rr5 | 128 | 128 | 1.00 |
| OPQ | hotpotqa | 0.95 | rr5 | None | None | not measured |
| OPQ | msmarco-sym | 0.80 | rr5 | 32 | 32 | 1.00 |
| OPQ | msmarco-sym | 0.90 | rr5 | 64 | 64 | 1.00 |
| OPQ | msmarco-sym | 0.95 | rr5 | 64 | 64 | 1.00 |
| OPQ | hotpotqa-sym | 0.80 | rr5 | 64 | 64 | 1.00 |
| OPQ | hotpotqa-sym | 0.90 | rr5 | 64 | 64 | 1.00 |
| OPQ | hotpotqa-sym | 0.95 | rr5 | 128 | 128 | 1.00 |
| OPQ | fiqa | 0.80 | rr5 | 32 | 32 | 1.00 |
| OPQ | fiqa | 0.90 | rr5 | 64 | 64 | 1.00 |
| OPQ | fiqa | 0.95 | rr5 | 64 | 64 | 1.00 |
| OPQ | nq | 0.80 | rr5 | 32 | 32 | 1.00 |
| OPQ | nq | 0.90 | rr5 | 64 | 64 | 1.00 |
| OPQ | nq | 0.95 | rr5 | 64 | 64 | 1.00 |
| OPQ | quora | 0.80 | rr5 | 32 | 32 | 1.00 |
| OPQ | quora | 0.90 | rr5 | 64 | 64 | 1.00 |
| OPQ | quora | 0.95 | rr5 | 64 | 64 | 1.00 |
