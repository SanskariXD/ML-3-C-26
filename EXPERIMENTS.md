# Experiment log

**Append one row per change, the moment you have the number. A change with no logged number
did not happen — revert it.**

This file is the project's memory. The baseline plateaued at 0.945 *blind* because its run kept
no metrics; nobody knew US and India differed until it was measured. Do not recreate that hole.

---

## How to log

Copy this template, fill it, append under **Log** (newest last). Keep it to a screenful.

```markdown
### YYYY-MM-DD — short title
- **Hypothesis:** the mechanism, in one sentence. Why should this help, given FINDINGS.md?
- **Change:** files touched / config knobs changed.
- **Command:** exact command, including --dev-frac.
- **Before → After:** holdout_stage2_f05 X → Y (**delta**)
  - blocking_recall A → B | oracle_f05 C → D | pairs_per_s1 E → F
  - per-country: US u1 → u2, India i1 → i2
- **Verdict:** KEEP / REVERT / INCONCLUSIVE — and why.
- **Notes:** surprises, side effects, follow-ups.
```

Always report `pairs_per_s1` alongside recall: candidate-set size is **explicitly graded**, so a
recall gain bought with a much larger candidate set may be a net loss in the final ranking.

## The promotion gate (before any full-scale run)

A full run costs 4–9 h. All four must hold:

1. Positive delta at `--dev-frac 0.03`.
2. Delta **survives** at `--dev-frac 0.3` (does not shrink toward zero).
3. Combined pending gain **≥ +0.005** on the dev slice.
4. You can state the **mechanism** in one sentence, consistent with the error taxonomy.

Dev-slice scores are optimistic (~0.984 where full scale gave 0.945). **Trust deltas, not
absolutes.**

## Metric reference

| Metric | Source | Baseline (dev 0.03) |
|---|---|---|
| `holdout_stage2_f05` | `work_dev/models/report.json` | 0.983787 |
| `holdout_blocking_recall` | same | 0.974226 |
| `holdout_oracle_f05` | same — **ceiling given candidates** | 0.990727 |
| `pairs_per_s1` | same — **graded, keep small** | 20.94 |
| `per_country_holdout_f05` | same | US 0.989606 / India 0.974928 |
| **Leaderboard** | Unstop portal | **0.945193** |

Targets: **0.986** = top 10. **~0.999** = measured ceiling.

---

# Log

### 2026-09-26 — BASELINE (reference row, do not overwrite)
- **Hypothesis:** n/a — establishing the reference.
- **Change:** none. Pipeline as inherited: normalize → IDF multi-key blocking (`k_key=40`) → 69
  features → 2-stage LightGBM (10 folds) → `expected_f` decision + greedy one-owner mask.
  **`--dense` was OFF.**
- **Command:** `src/run.py train --data-dir ../../student_resource/dataset --work-dir work_dev --dev-frac 0.03 --keep-intermediates`
- **Results:** `holdout_stage2_f05` **0.983787** | blocking_recall 0.974226 | oracle_f05 0.990727
  | pairs_per_s1 20.94 | US 0.989606 / India 0.974928 | stage-1 AUC 0.99994, stage-2 AUC 0.99996
  | decision `expected_f τ=0.02 γ=0.85 miss=0.0`
- **Full-scale leaderboard:** **0.945193** (the dev slice is optimistic by ~0.039)
- **Verdict:** KEEP as reference.
- **Notes:** Loss decomposes as **blocking 0.0093** + **matcher/decision 0.0069**.

### 2026-09-26 — Verified the generator's transformations against ground truth
- **Hypothesis:** the noise is a finite synthetic rule set, so alias tables should be mined, not
  hand-written.
- **Change:** added `src/verify_gt.py`. No pipeline change.
- **Command:** `src/verify_gt.py --sample work/sample.json`; `--one-owner .../train_ground_truth.tsv`
- **Results (139,271 true pairs):** street abbrev 29.1%, partial reorder 28.0%,
  **address UPPERCASED 53.35% of S2 vs 0.01% of S3**, number perturbed 15.7%, name truncated
  15.6%, diacritics 13.6%, legal-suffix dropped 13.5%, non-Latin name 6.7%, domain form 5.7%,
  transposition 5.6%, typo 5.5%, empty address 4.5%, **random name only 2.08%**.
  **Proven exactly on the full 2.08M entities / 7.64M records: one-owner holds, caps hold
  (n_S2 ≤ 5, n_S3 ≤ 6), zero violations.**
- **Verdict:** KEEP (evidence).
- **Notes:** Two corrections. "Appended descriptor words (Service/Center/Enterprises)" is only
  **0.94%** — hand-coding that list would have been useless. `address_UPPERCASED` is a strong,
  previously unnoticed **S2-vs-S3 discriminator** and is not yet exploited as a feature.

### 2026-09-26 — Error taxonomy of the trained model
- **Hypothesis:** find out where the 0.945 actually leaks instead of guessing.
- **Change:** added `src/error_analysis.py`, `src/profile_misses.py`.
- **Command:** `src/error_analysis.py --work-dir work_dev`; `src/profile_misses.py --work-dir work_dev`
- **Results:** 13,378 holdout entities, 1,627 (12.16%) with ≥1 error, 1,882 error items:
  **blocking_miss 63.1%**, missed_in_candidates 29.1%, fp_distractor 6.1%, fp_stolen 1.0%,
  fp_on_singleton 0.7% → **92.2% false negatives**.
  Miss traits (lift vs found): no shared name token 72.1% (5.5×), **non-Latin name 37.1%
  (6.1×)**, **empty address 23.0% (6.1×)**, **no shared token at all 0.08%**.
  Per-country miss mix: India blocking_miss 71.6%, US 51.0%.
- **Verdict:** KEEP (evidence). **Reorders the whole plan: blocking first, precision last.**
- **Notes:** **99.92% of missed pairs are reachable** — blocking had the signal and the
  top-K/`max_df` caps discarded it. `anyascii` transliterates phonetically (`पावर`→`pavara`),
  which never token-matches `power`; a mined token table is needed.

### 2026-09-26 — Decision-layer bounds (threshold work is exhausted)
- **Hypothesis:** better thresholds / per-country cuts / segment rules would gain score.
- **Change:** added `src/decision_analysis.py`. No pipeline change.
- **Command:** `src/decision_analysis.py --work-dir work_dev`
- **Results:** shipped rule 0.984071 | **best single global τ=0.69 → 0.983797 (−0.000274)** |
  **per-country τ → −0.000083** | oracle_topk 0.990587 (+0.006516) | oracle_subset 0.990727
  (+0.006655). Calibration by segment is tight (non-Latin 0.3080 actual vs 0.3064 predicted;
  empty-address 0.5609 vs 0.5670; plain 0.2968 vs 0.2969).
  Counts: mean predicted 3.344, mean oracle-k 3.369, mean true 3.462.
- **Verdict:** **REVERT the idea.** Threshold, per-country and segment tuning are all dead ends.
- **Notes:** `oracle_topk` ≈ `oracle_subset` (gap 0.00014) ⇒ **ranking is effectively solved**.
  The remaining ≈0.0065 is entirely **per-entity k selection** — a group-size prediction problem.
  True matches with empty addresses score `mean_p|y=1` 0.9106 vs 0.9943 for plain records; that
  is where marginal k decisions go wrong.

### 2026-09-26 — Repo restructure + smoke test (no score impact)
- **Change:** pipeline moved `README/` → `code/business_entity_resolution/` (the required
  submission layout); analysis scripts consolidated into its `src/`; 0.945 submission archived to
  `baseline/output_0.945193/`; added `tests/smoke_test.py` (was referenced but missing); fixed
  `package_submission.py` paths and added `work_dev` to its skip list.
- **Command:** `.venv/bin/python tests/smoke_test.py`
- **Results:** `SMOKE TEST PASS` — full pipeline on synthetic data, submission contract asserted.
- **Verdict:** KEEP. No model change; scores unaffected.

### 2026-09-26 — BM25 top-k blocking as a REPLACEMENT (literature-driven) — NEGATIVE
- **Hypothesis:** Sparkly (PVLDB 2023) reports top-k TF/IDF blocking beating 8 SOTA blockers.
  Our blocker *discards* keys above `max_df_pair=500`/`max_df_name=200` (0.16% of a 310k index),
  so pairs sharing only common tokens get no candidate; BM25 ranks instead of discarding, and
  99.92% of our missed pairs do share a token.
- **Change:** added `src/blocking_probe.py` (measurement only — no pipeline change).
- **Command:** `src/blocking_probe.py --work-dir work_dev`
- **Results:** BM25 alone, same slice/holdout: recall 85.03% @k=5, 91.30% @k=10, 93.08% @k=20,
  **94.29% @k=40** — versus incumbent **97.42% @ 20.94 cands/entity**.
- **Verdict:** **REVERT the idea. Do not replace blocking with BM25.**
- **Notes:** Mechanism for the failure: our signal lives in field *combinations* (house number ×
  rare street token); pooling all tokens into one bag lets common city/state tokens dominate.
  Published blocker rankings are benchmark averages — this dataset is not the average. Testing
  cost ~1 h and saved a costly rebuild.

### 2026-09-26 — UNION of incumbent + BM25 top-k — POSITIVE, accumulate
- **Hypothesis:** (Papadakis survey) union complementary key families. BM25 losing outright does
  not mean it fails on the *same* pairs as the incumbent.
- **Change:** `src/blocking_probe.py` extended to report union recall, candidate cost, rescue
  rate and the exact oracle F₀.₅ ceiling. Still measurement only.
- **Command:** `src/blocking_probe.py --work-dir work_dev`
- **Results (oracle ceiling, incumbent = 0.990727):**
  | k | union recall | cands/ent | rescued | oracle F₀.₅ | Δ |
  |---|---|---|---|---|---|
  | **5** | **98.18%** | **22.32** | **29.5%** | **0.993753** | **+0.003026** |
  | 10 | 98.50% | 26.01 | 41.6% | 0.994617 | +0.003890 |
  | 20 | 98.60% | 34.74 | 45.8% | 0.994932 | +0.004205 |
  | 40 | 98.72% | 53.43 | 50.4% | 0.995304 | +0.004577 |
- **Verdict:** **KEEP the direction — k=5. Do NOT promote to a full run alone.**
- **Notes:** k=5 is the knee: +0.003026 ceiling for +1.24 candidates/entity (+5.9%). Beyond k=10
  the trade collapses (k=40 = +154% candidates for +0.0015 more ceiling) and candidate size is
  graded. **This is a CEILING gain, not realized score**: the rescued pairs are the hard ones the
  incumbent could not reach, and extra candidates add false-positive risk, so expect **+0.002 to
  +0.003 realized** — below the +0.005 gate on its own. Next step is implementing the union in
  `blocking.py` and measuring *realized* `holdout_stage2_f05`.

### 2026-09-26 — Looser max_df caps (2000/800) — NEGATIVE / FLAT
- **Hypothesis:** raising `max_df_pair`/`max_df_name` recovers pairs whose only shared tokens are common (FINDINGS: 99.92% of misses share a token).
- **Change:** `--max-df-pair 2000 --max-df-name 800` (defaults 500/200). No BM25/dense.
- **Command:** `src/run.py train --work-dir work_exp_maxdf --dev-frac 0.03 --keep-intermediates --max-df-pair 2000 --max-df-name 800`
- **Before → After:** holdout_stage2_f05 0.983787 → **0.983762 (−0.000025)**
  - blocking_recall 0.974226 → 0.974183 | oracle_f05 0.990727 → 0.990706 | pairs_per_s1 20.94 → 21.33
  - per-country: US 0.989606 → 0.989655, India 0.974928 → 0.974791
- **Verdict:** **REVERT.** Top-K=40 still discards; looser df alone does not help on this slice.

### 2026-09-26 — k_key=60 — SMALL POSITIVE, costly candidates
- **Hypothesis:** raising top-K recovers more of the reachable misses.
- **Change:** `--k-key 60` (default 40).
- **Command:** `src/run.py train --work-dir work_exp_k60 --dev-frac 0.03 --keep-intermediates --k-key 60`
- **Before → After:** holdout_stage2_f05 0.983787 → **0.984196 (+0.000409)**
  - blocking_recall 0.974226 → 0.975629 | oracle_f05 0.990727 → 0.991226 | pairs_per_s1 20.94 → **26.13** (+25%)
  - per-country: US 0.989606 → 0.989745, India 0.974928 → 0.975747
- **Verdict:** **WEAK KEEP / prefer BM25.** Gain is real but tiny vs +24% candidates (graded).

### 2026-09-26 — BM25 union top-5 into blocking — KEEP (best so far)
- **Hypothesis:** union complementary BM25 top-5 with key blocking (measured oracle +0.003026).
- **Change:** implemented `bm25_k` in `blocking.py` / `config.py` / `run.py --bm25-k`; feature `bit_bm25`.
- **Command:** `src/run.py train --work-dir work_exp_bm25 --dev-frac 0.03 --keep-intermediates --bm25-k 5`
- **Before → After:** holdout_stage2_f05 0.983787 → **0.986418 (+0.002631)**
  - blocking_recall 0.974226 → **0.981824** | oracle_f05 0.990727 → **0.993753** | pairs_per_s1 20.94 → **22.18** (+5.9%)
  - per-country: US 0.989606 → **0.991420**, India 0.974928 → **0.978802**
- **Verdict:** **KEEP.** Realized gain matches the probe (+0.0026). Best single lever measured. Still below +0.005 gate alone — stack with `--dense` next.

### 2026-09-26 — `--dense` (e5-small, k_dense=15) alone — KEEP (new best)
- **Hypothesis:** multilingual-e5 embeds Devanagari+Latin; was OFF in every prior run; targets 37% non-Latin blocking misses.
- **Change:** `--dense` only (`bm25_k=0`). MPS encode on M5 Air; train resumed from cache after OOM mid-LightGBM.
- **Command:** `src/run.py train --work-dir work_exp_dense --dev-frac 0.03 --keep-intermediates --dense`
- **Before → After:** holdout_stage2_f05 0.983787 → **0.990239 (+0.006452)**
  - blocking_recall 0.974226 → **0.993373** | oracle_f05 0.990727 → **0.997942** | pairs_per_s1 20.94 → **31.10** (+48%)
  - per-country: US 0.989606 → 0.991871, India 0.974928 → **0.987754** (+0.0128)
- **Verdict:** **KEEP — new champion on the 3% slice.** Clears the +0.005 gate alone. Candidate cost is high (31/S1); next: stack BM25∪dense then CNP/meta-block to shrink pairs, and confirm at `--dev-frac 0.3`.

---

## Leaderboard (dev 0.03, dense OFF baseline = 0.983787)

| Config | F₀.₅ | Δ | pairs/S1 | Verdict |
|---|---|---|---|---|
| baseline | 0.983787 | — | 20.9 | ref |
| max_df 2000/800 | 0.983762 | −0.00003 | 21.3 | REVERT |
| k_key=60 | 0.984196 | +0.00041 | 26.1 | weak |
| BM25 k=5 | 0.986418 | +0.00263 | 22.2 | KEEP |
| `--dense` k=15 | 0.990239 | +0.00645 | 31.1 | KEEP |
| BM25 k=5 ∪ dense k=15 | 0.991003 | +0.00722 | 32.2 | KEEP |
| **+ non-Latin dense to rank 50** | **0.991332** | **+0.00754** | **34.2** | **BEST** |

---

### 2026-09-26 — BM25 top-5 ∪ dense k=15 — KEEP (new best)
- **Hypothesis:** the two retrievers miss different pairs, so the union raises the candidate ceiling above dense alone.
- **Change:** `--dense --bm25-k 5` together. Reused the 3% prepare + e5 embeddings from `work_exp_dense`. BM25 hit collection is vectorized; a 1.2 GB gemm chunk was slower than chunk 400 on 16 GB, so chunk stays 400. Do not export `OMP_NUM_THREADS=10` together with `OPENBLAS_NUM_THREADS=10` — that deadlocks LightGBM's OpenMP barrier on this Mac.
- **Command:** `src/run.py train --work-dir work_exp_stack --dev-frac 0.03 --keep-intermediates --dense --bm25-k 5 --dense-batch 256`
- **Before → After:** holdout_stage2_f05 0.983787 → **0.991003 (+0.007216)**; vs dense-alone 0.990239 (**+0.000764**)
  - blocking_recall 0.974226 → **0.994474** | oracle_f05 0.990727 → **0.998353** | pairs_per_s1 20.94 → **32.22**
  - per-country: US 0.989606 → **0.992319**, India 0.974928 → **0.988999**
  - stage-2 AUC still ~0.99997. Decision `expected_f τ=0.02 γ=0.6 miss=0.1`
- **Verdict:** **KEEP.** Best 3% number. Realized gain over dense is small because ranking was already solved; the oracle moved much more (0.9984), so the leftover is group-size selection, not another embedder.
- **Notes:** Cached resume of train-only was 1.4 min. Cold blocking on this slice was ~2.5 min (BM25 dominates).

---

## Next up (highest measured value first)

| # | Action | Why | Expected |
|---|---|---|---|
| 1 | ~~`--dense`~~ | **DONE: +0.00645** | KEEP |
| 2 | ~~BM25 top-5~~ | **DONE: +0.00263** | KEEP |
| 3 | ~~Stack BM25 ∪ dense~~ | **DONE: 0.991003** | KEEP |
| 4 | Confirm stack at `--dev-frac 0.3` | promotion gate | must survive scale |
| 5 | Predict n_S2 / n_S3 per entity | oracle 0.9984 vs realized 0.9910 | up to ~+0.007 |
| 6 | Meta-blocking / lower `k_dense` | pairs 32 is graded | size ↓ |
| 7 | Granite-embedding-97m-multilingual-r2 | Apache, higher MTEB than e5-small; **unmeasured here** | maybe recall ↑ |

**Winner so far:** **BM25∪dense∪script@50∪empty-BM25@40∪housenum p-adjust** (0.991732).

### 2026-09-26 — Winning stack is now the default
- **Change:** `run.py all` uses `dense=True`, `bm25_k=5`, `k_dense_script=50` unless overridden. `--no-dense` turns embeddings off. Smoke test still passes `--no-dense --bm25-k 0 --k-dense-script 0`.
- **Score:** no new measurement. This is the config that scored **0.991332** on the 3% slice (`work_exp_script` plus the BM25 union already inside that run).
- **Verdict:** KEEP. A full `run.py all` now trains that pipeline.

### 2026-09-26 — Deeper BM25 for empty-address index rows — KEEP (small)
- **Hypothesis:** residual blocking misses with an empty address have median BM25 rank **33** and ~55% sit inside rank 50; `bm25_k=5` keeps none of them. Dense cannot save them (median dense rank 184).
- **Change:** `bm25_k_empty=40` keeps BM25 rank&lt;5 always, and ranks 5–39 when the *index* address is empty. Default on. `--bm25-k-empty 0` disables.
- **Command:** `src/run.py train --work-dir work_exp_bm25_empty --dev-frac 0.03 --keep-intermediates --bm25-k-empty 40` (embeddings reused from `work_exp_script`)
- **Before → After:** holdout_stage2_f05 0.991332 → **0.991548 (+0.000216)**
  - blocking_recall 0.995359 → **0.996762** | oracle_f05 0.998741 → **0.999062** | pairs_per_s1 34.17 → **36.79**
  - blocking_miss errors 215 → **150**; empty-addr share of misses 54% → **35%**
  - per-country: US 0.992385 → **0.992589**, India 0.989728 → **0.989962**
- **Verdict:** **KEEP.** Real, small. Perfect selection is now **0.999**; realized score is still stuck on mid-p near-copies (`missed_in_candidates` is 67% of errors). **1.0 is not reachable** on this slice: the decision gap to oracle-topk is still ~0.007 and no threshold/heuristic closed it.
- **Notes:** Ceiling clarification — oracle_subset **0.999062**. The leftover to 1.0 is mostly choosing which p≈0.4 near-copy is true, not more blocking.

### 2026-09-26 — House-number boost/pen on calibrated p — KEEP (small)
- **Hypothesis:** selected FPs have median |Δ first house number|=5; true matches are exact 77% of the time. Boost exact, penalize close-but-unequal.
- **Change:** before `expected_f`, `p += 0.05` on exact first-`a_nums` match, `p -= 0.05` when 1≤|Δ|≤5. Defaults `housenum_boost=0.05`, `housenum_pen=0.05`. Stored in `decision` meta for predict.
- **Command:** `src/run.py train --work-dir work_exp_housenum --dev-frac 0.03 --keep-intermediates` (pairs/features reused from `work_exp_bm25_empty`)
- **Before → After:** holdout_stage2_f05 0.991548 → **0.991732 (+0.000184)**
  - India 0.989962 → **0.990355**, US 0.992589 → **0.992637** | pairs/oracle unchanged
- **Verdict:** **KEEP.** Decision-only, no candidate growth. Stacked with empty-addr BM25.

### 2026-09-26 — Group-size head and exact-name force-include — REVERT
- **Hypothesis:** the +0.0069 oracle-prefix gap is predictable from the score profile, and unique exact names at p≈0.4 are true matches the decision drops.
- **Change:** none kept. Scores cached at `work_exp_stack/train/scores_all.npz`.
- **Results:** shipped 0.991003. LightGBM predicting oracle-k from the top-8 probabilities → **0.990279 (−0.000724)**. `round(sum p)` 0.989281. Forcing a unique `n_tset≥0.99` pair in → **0.990699 (−0.000304)**, 39 adds of which only 12 were true. Missed true pairs have median p **0.40** and median name-token overlap **1.0**, the same name overlap as the 173 selected false matches.
- **Verdict:** **REVERT.** The leftover is which mid-score near-copy is true. The profile does not say. Raw holdout (953 / 13,452 entities, 1,018 errors): missed-in-candidates 587, blocking_miss 256, fp_distractor 140, fp_stolen 20, fp_on_singleton 15. Examples are house-number-off-by-a-few and empty-address copies on both sides of the label.

### 2026-09-26 — Deeper dense neighbors for non-Latin names — KEEP (small)
- **Hypothesis:** residual blocking misses whose name is non-Latin have median dense rank 29 (66% inside rank 50); k=15 keeps none of them.
- **Change:** `k_dense_script=50` keeps rank&lt;15 always, and ranks 15–49 when either name has no Latin letter. `--k-dense-script 50`.
- **Command:** `src/run.py train --work-dir work_exp_script --dev-frac 0.03 --keep-intermediates --dense --bm25-k 5 --k-dense-script 50`
- **Before → After:** holdout_stage2_f05 0.991003 → **0.991332 (+0.000329)**
  - blocking_recall 0.994474 → **0.995359** | oracle_f05 0.998353 → **0.998741** | pairs_per_s1 32.22 → **34.17**
  - per-country: US 0.992319 → 0.992385, India 0.988999 → **0.989728**
- **Verdict:** **KEEP.** Real, small, and it pushes the perfect-selection ceiling over 0.998. It does not move the realized score to 0.998: that still requires knowing which p≈0.4 near-copy is the true one.
- **Notes:** Torch and LightGBM deadlock on this Mac if they share a process. `run.py` re-execs after blocking so training starts clean. Embeddings and the score matrix stay cached.

### 2026-09-26 — Cross-encoder rerank ceiling, model held fixed — do not train it
- **Hypothesis:** a Ditto-style reranker on the shortlist would cut false positives, which zero a singleton and dilute F₀.₅.
- **Change:** none. `decision_analysis.py` plus a shortlist probe on `work_exp_stack`.
- **Command:** `src/decision_analysis.py --work-dir work_exp_stack`
- **Before → After:** shipped holdout **0.990906** (analysis mask; report.json is 0.991003)
  - best global τ=0.73 → **−0.000381** | per-country τ → **−0.000366**
  - oracle_topk **0.997831 (+0.006925)** | oracle_subset **0.998353 (+0.007447)**
  - scorer gap (subset − topk) **+0.000522** — that is all a reranker can gain by reordering
  - selected pairs: 45,483 true, **175 false**, **15** of them on true singletons
  - perfect drop of those 175 FPs → **0.994666 (+0.003760)**. An emb_cos × name-token gate could not beat the shipped rule (FP median emb 0.934 and p 0.894, almost on top of the true matches)
- **Verdict:** **REVERT the idea.** Do not fine-tune the cross-encoder. The FPs are near-copies, not junk the reranker can see and the trees cannot. The larger hole is choosing k per entity (+0.0069), which a shortlist reranker does not do.

### 2026-09-26 — France proxy: train and tune on US only, score India — KEEP as evidence
- **Hypothesis:** France is 15.0% of test S1 (259,452 / 1,732,544) and has no labels. India left out of both training and decision tuning is the measurable stand-in.
- **Change:** none to the submission model. `--train-countries us` in `work_exp_loco_us`, then isotonic + `expected_f` refit on US folds only.
- **Command:** `src/run.py train --work-dir work_exp_loco_us --dev-frac 0.03 --keep-intermediates --dense --bm25-k 5 --train-countries us`
- **Before → After:** India holdout 0.988999 (in training) → **0.979812** (never in train or tune), **−0.0092**. US holdout stays **0.992313**.
- **Verdict:** **KEEP (evidence).** An unseen country does not collapse and does not need a looser threshold — that is what creates singleton false positives. French street types and legal suffixes (`rue`, `sarl`, `sas`) are already in `normalize.py`. Keep the US+India decision rule for France. A France-only stricter cut has no label to tune on; the transfer gap, spread over 15% of test S1, is about **0.001** on the leaderboard, smaller than the group-size hole.

### 2026-09-26 — Domain/acronym NC keys — REVERT
- **Hypothesis:** initials + TLD-stripped squash on NC would recover domain-form misses.
- **Change:** expanded NC key variants; reverted in `blocking.py`.
- **Command:** `src/run.py train --work-dir work_exp_domain --dev-frac 0.03 --keep-intermediates`
- **Before → After:** holdout_stage2_f05 0.991732 → **0.991116 (−0.000617)**
  - pairs_per_s1 36.79 → **39.52** | blocking_recall flat | oracle flat/down
- **Verdict:** **REVERT.** More pairs, no recall gain, score down.

### 2026-09-26 — Runtime: unique-E5 + resume, FAISS (CUDA), fuzzy brank gate — KEEP
- **Hypothesis:** dedup texts before E5, crash-resume encode, FAISS exact kNN, and RapidFuzz only for `brank<24` (+ dense hits) cut wall clock without hurting recall.
- **Change:** `dense.py` unique-text encode + `.progress` resume; FAISS `IndexFlatIP` on CUDA/CPU (torch on MPS); `feat_fuzz_brank_max=24` default; `faiss-cpu` in `requirements-dense.txt`.
- **Command:** `src/run.py train --work-dir work_speed --dev-frac 0.03 --keep-intermediates --dense-batch 256`
- **Before → After:** holdout_stage2_f05 0.991732 → **0.991988 (+0.000256)**
  - blocking_recall 0.996762 | oracle_f05 0.999062 | pairs_per_s1 **36.79** (unchanged)
  - India 0.990355 → **0.990983**, US 0.992637 → **0.992648**
  - Timings (3% Mac): E5 dedup only **0.5%** unique savings (not a win here); encode resume **works**; features India 1.27M pairs **11s** (fuzzy gate); cached resume train **1.9 min** end-to-end after embs exist
- **Verdict:** **KEEP.** Score flat/up; features much faster. Dedup is still worth keeping for crash-resume + full-scale dupes; don’t expect big encode savings on this generator. FAISS is for GPU boxes, not MPS.

### 2026-09-26 — A100 readiness: hard-neg subsample, adaptive dense (opt-in), telemetry
- **Hypothesis:** teammate checklist — keep accuracy stack; add hard-neg training cap, optional adaptive dense-k, resource logs; don’t rebuild entity-level k (already REVERT).
- **Change:** `_subsample_hard_neg` (all positives + hardest bscore/dcos/brank negatives); `dense_adaptive` + `k_dense_easy/empty` (default **off**); `timed()` logs cpu% + gpu mem only if torch already loaded (fixed infinite re-exec when `_gpu_stats` imported torch); `scripts/run_a100.sh`.
- **Already true (no change):** stage caches, mmap features, pair dedup in `merge_extra`, fuzzy gate@24, fold crc, max_train_rows=30M, script/empty BM25 adaptive.
- **Command:** `tests/smoke_test.py` → **PASS**. Adaptive / hard-neg full-slice measure: **pending** (`--dense-adaptive` on `--dev-frac 0.03` before promote).
- **Verdict:** **KEEP plumbing.** Do not enable `--dense-adaptive` on A100 until 3% then 0.3 measure. Entity-level k selection stays off.

### 2026-09-26 — Multi-GPU E5 encode (T4×2) — KEEP plumbing, score N/A
- **Hypothesis:** Kaggle T4×2 can nearly double unique-text encode throughput by sharding across devices; overall wall clock only partially improves (features/LGB stay CPU).
- **Change:** `dense.py` loads one `SentenceTransformer` per CUDA device and encodes unique-text waves in parallel (`ThreadPoolExecutor`); single-GPU / MPS / CPU path unchanged. Crash-resume progress still advances contiguously per wave.
- **Command:** not measured on a holdout slice yet (live full Kaggle run already in flight on 1×T4 — do not interrupt).
- **Before → After:** score **N/A** (encode-only speed). Expect log line `multi-GPU encode: 2 CUDA devices` + `ngpu=2` on next encode.
- **Verdict:** **KEEP.** Applies automatically when `torch.cuda.device_count() > 1`.
