# False-clear rate — how often "good enough" was wrong

The [rank certificate](certification.md) ships a distribution-free *floor* on rank
agreement. This is its empirical companion. Where the certificate bounds the
worst case, `false_clear` measures the actual case: **how often does a cheap
nominal metric approve a compression that your consumer then rejects?**

It is the number behind the KV-keys finding. A per-channel key compression can
hold cosine similarity at 0.995 and still send perplexity to ~1e4. Cosine
cleared; the consumer failed. That is a *false clear*, and this module measures
its rate instead of leaving it as an anecdote.

## What it is

```python
from turboquant_pro import false_clear, false_clear_from_scores

# from two per-item boolean outcomes
report = false_clear(nominal_accept, consumer_ok)
report.false_clear_given_cleared  # P(consumer fails | nominal cleared) -- the headline
report.false_clear_rate           # P(nominal cleared AND consumer failed) -- joint
report.conservative_miss_rate     # P(nominal rejected but consumer was fine) -- harmless
report.verdict                    # "ok" / "warn" / "fail"
```

The report is **directional**, matching the runtime policy's one-sided design: a
*false clear* is the harmful direction (you were reassured, the consumer broke); a
*conservative miss* is the harmless one (you flagged a result the consumer would
have accepted). The verdict gates only on the harmful direction, on
`P(consumer fails | nominal cleared)` — the untrustworthiness of a "clear."

## From scores

More often you have continuous scores and a threshold on each — a cheap nominal
score (cosine, reconstruction MSE) and an expensive consumer score (recall@k,
negative perplexity, a task metric):

```python
report = false_clear_from_scores(
    nominal=cosine_per_item,          # cheap metric
    consumer=recall_per_item,         # true consumer outcome
    nominal_threshold=0.9,            # accept when cosine >= 0.9
    consumer_threshold=0.95,          # consumer ok when recall >= 0.95
    # for a distance / loss, pass *_higher_is_better=False
)
```

## The certificate and the false-clear rate together

They answer different questions and are strongest read side by side:

| | certificate (`tau_floor`) | `false_clear` |
|---|---|---|
| kind | distribution-free floor | empirical rate |
| answers | worst-case rank distortion | how often a "clear" was actually wrong |
| a loose floor | says "certify nothing" (`vacuous`) even when practice is fine | may still be near zero |
| a false clear | not visible until you measure the consumer | is exactly this number |

A `vacuous` certificate with a near-zero false-clear rate means the floor is
loose but practice is fine. A non-vacuous certificate is a guarantee; a low
false-clear rate is evidence. Use the floor to decide whether exact reranking is
*required*; use the false-clear rate to decide whether your nominal accept-metric
can be *trusted* on this consumer.

## Honest scope

The thresholds (`FALSE_CLEAR_WARN = 0.05`, `FALSE_CLEAR_FAIL = 0.20`) are
conventions on the conditional rate, not measurements — no sweep here has fixed
the operational knee for a given consumer. Pass your own `warn` / `fail` once you
have calibrated them. The rate is only as meaningful as the `consumer_ok` you feed
it: it measures agreement with a consumer outcome you define, and says nothing
about a consumer you did not evaluate.
