# Literature review — what transfers to this problem, and what does not

> Every claim below was **tested on our own data** before being recommended. Where a
> published finding failed to transfer, that is recorded too — those are the expensive
> mistakes this file exists to prevent.

## Sources

| Paper | Venue | Relevance |
|---|---|---|
| Paulsen, Govind & Doan — [*Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching*](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) | PVLDB 16(7):1507–1519, 2023 | top-k TF/IDF blocking beats 8 SOTA blockers; authors recommend top-k because it improves recall |
| Papadakis, Skoutas, Thanos & Palpanas — [*Blocking and Filtering Techniques for Entity Resolution: A Survey*](https://arxiv.org/abs/1905.06167) | ACM CSUR, 2020 | schema-agnostic blocking reaches higher recall than schema-based; unioning key families is standard practice |
| Zeakis, Papadakis, Skoutas & Koubarakis — [*Pre-trained Embeddings for Entity Resolution: An Experimental Analysis*](https://www.vldb.org/pvldb/vol16/p2225-skoutas.pdf) | PVLDB 16(9):2225–2238, 2023 | 12 language models × 17 benchmarks for blocking and matching |
| Brinkmann et al. — [*SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines*](https://arxiv.org/abs/2303.03132) | 2023 | learned blocking producing candidate sets ~half the size of alternatives |
| Papadakis et al. — [*Meta-Blocking: Taking Entity Resolution to the Next Level*](http://dit.unitn.it/~themis/publications/tkde13-metablocking.pdf) | TKDE, 2014 | prune a block collection via block-filtering and edge-pruning (CEP/CNP/WEP) |
| Thirumuruganathan et al. — *DeepBlocker* / Wu et al. — [*Blocker and Matcher Can Mutually Benefit*](https://vldb.org/pvldb/vol17/p292-wu.pdf) | PVLDB 2021 / 17(2), 2023 | deep and co-learned blocking |

*Content rephrased and summarised for licensing compliance.*

---

## Tested claim 1 — "top-k TF/IDF blocking is a strong baseline" (Sparkly)

**Why it looked compelling here.** Our blocker matches exact keys, then **discards** any key
whose document frequency exceeds `max_df_pair=500` / `max_df_name=200` — on a 310k-record index
that is 0.16% / 0.06%. A true pair whose only shared tokens are common ones therefore receives
no candidate at all. BM25 top-k never discards: a common term simply earns little score, so the
pair still surfaces when nothing better competes. `FINDINGS.md` measured that **99.92% of missed
pairs do share a token** — precisely the population this difference should address.

**Result: the claim did NOT transfer.** Schema-agnostic BM25 top-k, alone, is clearly worse
(`src/blocking_probe.py`, dev slice, same holdout):

| k | BM25 recall | cands/entity |
|---|---|---|
| 5 | 85.03% | 5.00 |
| 10 | 91.30% | 10.00 |
| 20 | 93.08% | 20.00 |
| 40 | 94.29% | 40.00 |
| **incumbent** | **97.42%** | **20.94** |

BM25 needs 40 candidates to reach 94.3%, where the incumbent gets 97.4% with 21. **Do not
replace blocking with BM25.** The reason is intelligible: our discriminative signal is
concentrated in field *combinations* (house number × rare street token), and pooling every token
into one bag dilutes it — common city/state tokens dominate the bag. The incumbent's compound
keys are simply more precise for this data. Published blocker rankings are averages over
benchmarks; this dataset is not the average.

## Tested claim 2 — "union complementary key families" (Papadakis survey) ✅ **TRANSFERS**

The right question was never "is BM25 better" but "does BM25 find pairs the incumbent *misses*".
It does, and cheaply:

| k | union recall | cands/entity | incumbent misses rescued | **oracle F₀.₅** | vs incumbent |
|---|---|---|---|---|---|
| — | 97.42% | 21.08 | — | 0.990727 | — |
| **5** | **98.18%** | **22.32** | **29.5%** | **0.993753** | **+0.003026** |
| 10 | 98.50% | 26.01 | 41.6% | 0.994617 | +0.003890 |
| 15 | 98.57% | 30.27 | 44.4% | 0.994847 | +0.004120 |
| 20 | 98.60% | 34.74 | 45.8% | 0.994932 | +0.004205 |
| 40 | 98.72% | 53.43 | 50.4% | 0.995304 | +0.004577 |

**k=5 is the knee: +0.003026 on the oracle ceiling for +1.24 candidates/entity (+5.9%).** Past
k=10 the trade collapses — k=40 costs +154% candidates for only +0.0015 more ceiling, and
candidate-set size is **explicitly graded**.

The two blockers fail on *different* pairs, which is exactly the survey's argument for unioning
schema-agnostic with schema-based blocking.

### What this is and is not

It is a **ceiling** gain, not a realized score gain. The model currently realizes 0.9838 of a
0.9907 ceiling. The rescued pairs are by construction the *hard* ones the incumbent could not
reach, so realization will be lower than on average pairs, and extra candidates also create
extra false-positive opportunity. **Honest expectation: +0.002 to +0.003 realized at k=5.**

Per the promotion gate in `AGENTS.md`, that is **below +0.005 on its own** — accumulate it with
other wins rather than triggering a full run for it alone.

## Untested but literature-supported (ranked)

1. **Embedding choice for the `--dense` pass.** Zeakis et al. (PVLDB 2023) find **S-GTR-T5**
   strongest on English ER blocking benchmarks. That model is English-centric (~1B, Apache-2.0)
   and does **not** address our Devanagari miss cluster. Keep `multilingual-e5-small` (MIT, 118M)
   — it is the right family for this data. Turning `--dense` on at all is still the higher-value
   test; swapping model size (e5-base) is a later A/B.
2. **Hybrid sparse ∪ dense, not score fusion.** IR practice (RRF / hybrid search) and our own
   BM25-union measurement agree: when retrievers fail on *different* pairs, **union the
   candidate sets**. Do not blend BM25/IDF/cosine into one score before top-K — incompatible
   scales. Cap the union (graded `pairs_per_s1`).
3. **Meta-blocking edge pruning** (CEP/CNP/WEP) to *shrink* the candidate set at fixed recall.
   Run this *after* recall expansions (BM25 ∪ dense ∪ adaptive df), not before. CNP (top-k
   edges per node) fits our per-S1 budget best.
4. **SC-Block** (learned contrastive blocking) — larger effort; defer until the cheap union +
   dense + mined-table stack is measured.

## Advanced research addendum (2026-09-26) — new probes + transfer notes

### A. Mined Indic→English token table (Sinha / MINT / XLEnt-style) — **transfers, method refined**

Literature (Sinha ACL 2009; MINT; XLEnt LSP-Align) mines name transliterations from parallel
text via **alignment**, not phonetic romanisation. Our generator is exactly that parallel
corpus.

**Measured on `work/sample.json` (10k cross-script true pairs):**

| Probe | Result |
|---|---|
| Space-split positional align (equal token count) | **10 000 / 10 000** pairs align 1:1 — generator preserves whitespace token cardinality |
| High-precision mappings (count≥5, 2× runner-up) | **886** script→English entries |
| Critical failure of `anyascii` | `प्राइवेट` → `praivet` (never matches `private`); `मीडिया`→`midiya`, `पावर`→`pavr` |
| Correct mined maps | `प्राइवेट`→`private`, `मीडिया`→`media`, `पावर`→`power`, plus Kannada/Telugu/Tamil/Bengali/Gujarati/Malayalam legal-suffix variants |

**Implementation rule:** mine on **original-script space tokens** before `anyascii`; inject the
English side into blocking name keys (`n_core` / NP/NC). Do **not** use Python `\w` on Indic
text — virama/matras shatter words into single aksharas (verified: `स्मार्ट` → 4 junk tokens).

**Do not import ParaNames / web dictionaries** (fair-play: no external lookup). GT-mined tables
only.

### B. Group-size prediction (the real 0.0065 decision headroom)

`decision_analysis.py` closed thresholds: best global τ and per-country τ are **flat/negative**.
`oracle_topk ≈ oracle_subset` ⇒ ranking is solved; the gap is **choosing k per entity**.

This is the plug-in cardinality problem (F-measure maximisation literature), not Magellan's
global threshold grid. Concrete recipe consistent with our caps:

1. Features from the calibrated score *profile*: top-11 probs, successive gaps, entropy,
   `#p>0.9/#p>0.5`, empty-address / non-Latin flags, `address_UPPERCASED` (S2 vs S3).
2. Predict `(n_S2, n_S3)` (two small heads or one multi-output LightGBM), clip to **5 / 6**.
3. Take source-aware top-n from the p-sorted list (already one-owner masked).

Expected: up to **+0.006** if prediction is accurate; start with a cheap probe that replaces
`expected_f` with `top-k*` where `k* = round(sum p)` clipped — that is the first ablation.

### C. Hard constraints still under-used

- One-owner: already greedy in `decide.exclusive_mask`.
- Caps: baseline submission had **348 entities** with n_S2>5 or n_S3>6 — free false positives.
  Clip after decision regardless of scores.
- `address_UPPERCASED` (53% S2 / 0% S3) is not just “unused” — `norm_addr` lowercases via
  `_ascii`, so the signal is **destroyed before featurization**. Compute it from the **raw**
  address (e.g. fraction of alpha chars that are uppercase) and pass it as a feature; do not
  try to recover it from normalized fields.
- `n_translit` / `n_phon` exist, but translit is only a **boolean flag** and phon cannot bridge
  Devanagari↔English lexical mismatch — blocking keys never see original-script tokens.

### D. Cross-encoder reranker (`rerank.py`) — **deprioritise**

Already coded (Ditto-style e5 cross-encoder on the shortlist). It can only reshuffle pairs the
decision already selected → **precision-only**. Precision is 7.8% of errors and stage-2 AUC is
0.99996. Do not spend GPU time here until blocking recall moves.

### E. France / address aliases — secondary, unsupervised OK

India (labelled) already trails US by 0.015 on blocking recall; fix Indic blocking first. For
France: mine street-type / region↔department aliases from **test-file co-occurrence** (allowed;
no external gazetteer). Synthesis of French labels remains optional after India moves.

### F. Stacking plan that can clear the +0.005 promotion gate

| # | Lever | Status | Expected Δ (dev slice) |
|---|---|---|---|
| 1 | Union BM25 top-5 into `blocking.py` | measured ceiling +0.003; **not implemented** | +0.002–0.003 realized |
| 2 | `--dense` (e5-small) | implemented, **never run** | recall ↑ (non-Latin) |
| 3 | GT-mined Indic→English table into name keys | method validated above | recall ↑ (India) |
| 4 | Adaptive `max_df` + split name/addr K budgets | not implemented | recall ↑ (df-cap misses) |
| 5 | Meta-blocking CNP after expansions | not implemented | `pairs_per_s1` ↓ |
| 6 | Group-size head + hard cap clip | not implemented | up to +0.006 |

Items 1–4 are complementary recall plays; measure **together** on `--dev-frac 0.03`, then
confirm at `0.3` before any full run.

## Reproduce

```bash
cd code/business_entity_resolution
.venv/bin/python src/blocking_probe.py --work-dir work_dev      # ~5 min
```
