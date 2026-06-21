# Repositioning the PCA-Matryoshka paper after the TAI desk-reject

## What actually happened
TAI-2026-Apr-A-00731 was **desk-rejected on scope, not quality.** The editor
called it "a decent work," **never sent it to review**, and bounced it because
IEEE Trans. AI is a *broad interdisciplinary* AI journal and judged
embedding/vector-DB compression to be a "specialized topic explored using AI
tools." Takeaway: **venue-fit failure, not a technical one.** Do not change the
science — re-aim it and re-frame the pitch.

## Core repositioning: from "ML method" to "data-systems contribution"
The paper currently leads as a machine-learning method ("dimension reduction for
non-Matryoshka embedding models"). At the venues where it *fits* — **data
management & information retrieval** — lead instead with the systems/data payoff:
training-free compression that makes billion-scale vector search fit a fraction
of the storage/RAM, validated with a rigorous retrieval-quality evaluation. Same
content, data-venue framing.

Concrete changes:
- **Title** → lead with the vector-DB/retrieval payoff. e.g.
  *"Training-Free Vector-Database Compression: PCA-Matryoshka Truncation and
  Scalar Quantization at 27x with 99.8% Recall@10."*
- **Abstract** → open on the deployment/cost problem (billion-scale vector stores
  are expensive; most deployed embedding models can't be truncated), not on
  Matryoshka *training*. The current abstract's 2nd–3rd sentences already say
  this — promote them to the lead.
- **Keywords** → vector databases, approximate nearest neighbor, embedding
  compression, information retrieval, data management. Drop "representation
  learning / intelligent systems" (those signal broad-AI, which got it rejected).
- **Promote the methodological finding** → "cosine similarity is *not* a reliable
  proxy for retrieval quality at high compression" is exactly the
  evaluation-rigor result IR/data reviewers reward. Make it a *named*
  contribution, not a footnote.
- **Drop the "Impact Statement"** — that's a TAI-only requirement; data venues
  don't use it.
- **Strengthen the systems story** → add storage/RAM and end-to-end retrieval
  latency/QPS for a real index (pgvector / FAISS), not just embedding-level
  cosine/recall. Data-venue reviewers want the *system* measured, and
  turboquant-pro already has the pgvector/FAISS connectors to produce it.

## Where to send it (ranked; dates as of June 2026 — verify before submitting)

1. **PVLDB / VLDB 2027, Research Track — TOP PICK.** Vector-DB compression is
   dead-center scope, and PVLDB uses **rolling monthly deadlines** (submit by the
   1st of a month), so you can submit in *weeks*, not wait a year. ~12pp,
   double-blind; conf Aug 2027, Athens.
   <https://vldb.org/2027/call-for-research-track.html>
2. **IEEE TKDE (journal).** Rolling submission; data-engineering scope (vector
   search, compression). No deadline pressure; best for a thorough version.
3. **ICDE 2027.** IEEE data-engineering flagship; check the 2nd-round deadline
   (~Oct 2026). Copenhagen, May 2027. Strong systems fit.
   <https://icde2027.github.io/cf-research-papers.html>
4. **CIKM 2027 (~May 2027 deadline).** IR + data + knowledge management; the
   retrieval-quality + evaluation-methodology angle fits well; shorter paper.
5. **SIGIR / ECIR.** If you lead hard with the *retrieval-quality and
   cosine-isn't-a-proxy* evaluation contribution. SIGIR full-paper deadline
   ~Jan 2027.
6. **MLSys 2027 (~Oct 2026).** If reframed toward systems-for-ML efficiency (the
   training-free pipeline + autotune + connectors).

**Avoid:** another broad AI venue — TAI again, AIJ, JAIR — and, despite the
editor's suggestion, **TPAMI**: it is also broad and brutally selective, and the
topical fit (a data-systems compression tool) is weak. The editor's "specialized
journal" advice points at data/IR venues, not another generalist one.

## The TechRxiv preprint is an asset (handle it right)
The paper is posted on **TechRxiv** (IEEE's preprint server) — a genuine positive,
not a consolation prize:
- **Priority / timestamp.** It establishes precedence on the PCA-Matryoshka idea
  and the cosine-isn't-a-proxy finding, with a citable DOI.
- **Not "prior publication."** Preprints are *not* peer-reviewed publications;
  PVLDB, ICDE, TKDE, CIKM, and SIGIR all explicitly permit prior preprints, so it
  does **not** block submission to any venue on the shortlist.
- **Double-blind handling.** PVLDB/ICDE/SIGMOD are double-blind. Keep the preprint
  posted (allowed), but in the submitted PDF refer to it in the third person and
  don't cite/link it in a way that reveals authorship; don't actively publicize
  the submission during review. This is standard and low-risk.
- **After acceptance,** update the TechRxiv version to point to the version of
  record (DOI), so citations converge on the published paper.

Action: add the TechRxiv DOI to `references.bib` and cite it as the preprint of
record; the AIoT paper can also cite it for the embedding-compression result.

## Don't self-overlap with the AIoT paper
- **AIoT paper** = the *system on the edge* (weights + KV + embeddings, device
  budget, energy/token on a Jetson Nano).
- **This paper** = the *embedding-compression method + vector-DB evaluation* for
  server/datacenter-scale retrieval.

They share the 27x / 99.8% recall@10 embedding result: keep it attributed to
*this* paper as the source of record, and have the AIoT paper cite it. Different
venues, different leading contributions — no dual-submission concern.

## Recommended immediate next step
Reframe for **PVLDB** (rolling deadline = fastest), retitle + re-lead the
abstract, add a pgvector/FAISS end-to-end retrieval-cost table, and submit on the
next monthly cutoff. It is simultaneously the best-fit and the fastest path off
the rejection.
