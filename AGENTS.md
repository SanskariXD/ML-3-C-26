# Agent handoff — read this first, completely, before touching anything

You are picking up a **live competition submission** that already scores **0.945193**. Top 10
requires **0.986**. The measured ceiling is **~0.999**.

There is a working pipeline, a verified measurement harness, and a diagnosis backed by
ground-truth evidence. Your job is to improve the score **without breaking what works** and
**without burning a day on a full-scale run that was never justified.**

---

## The two rules that matter most

### Rule 1 — Log every change in `EXPERIMENTS.md`. No exceptions.

One row per experiment, appended as soon as you have the number: date, what you changed, the
command, the metric before, the metric after, the delta, and your verdict (keep / revert /
inconclusive). **A change with no logged number did not happen and must be reverted.**

This exists because the baseline plateaued at 0.945 *blind*: its run kept no `report.json`, so
nobody knew the per-country split until we measured it. Do not recreate that hole.

### Rule 2 — Never run the full pipeline on a hunch.

A full-scale run is **4–9 hours** (see `HARDWARE.md`). Earn it.

**Promotion gate to a full run — all four must hold:**
1. Measured on the dev slice (`--dev-frac 0.03`, ~3 min) with a **positive delta**.
2. Confirmed at `--dev-frac 0.3` (~30 min) — the delta **survives scale** and does not shrink
   toward zero.
3. The combined gain of everything pending is **≥ +0.005 F₀.₅** on the dev slice. Below that,
   keep accumulating; scale noise will eat it.
4. The improvement has a **mechanism you can state in one sentence**, consistent with the
   error taxonomy in `FINDINGS.md`. "The number went up" is not a mechanism — it is often
   variance or a leak.

If a change is negative or flat on the dev slice, **revert it and log it as tried.** Negative
results are valuable; they stop the next person repeating you.

---

## What is already known — do not re-derive this

Read `FINDINGS.md` in full. It is measured, reproducible, and supersedes every other doc.
The short version:

**The error taxonomy (real model, real holdout, real ground truth, 1,882 errors):**

| Category | Share |
|---|---|
| `blocking_miss` — true match never became a candidate | **63.1%** |
| `missed_in_candidates` — was a candidate, not selected | **29.1%** |
| `fp_distractor` + `fp_stolen` + `fp_on_singleton` | 7.8% |

**92.2% of errors are false negatives.** Despite F₀.₅ being precision-weighted, precision is
*not* where the work is. The scorer is near-perfect (stage-2 AUC 0.99996), so the decision rule
is already conservative enough that almost nothing over-merges.

**Proven structural facts** (exact, across all 2,083,574 entities / 7,638,365 matched records):
- **One-owner holds**: every S2/S3 record belongs to at most one S1. Zero violations.
- **Caps hold**: n_S2 ≤ 5 and n_S3 ≤ 6. Zero violations.
- Every S1 entity is unique by normalized name+address (0 collisions).

**Blocking misses are recoverable, not irreducible**: 99.92% of missed pairs share ≥1 token
with their S1 record. Top-K=40 and the `max_df_pair=500` / `max_df_name=200` caps are discarding
them. Misses concentrate in non-Latin names (37%, 6.1× lift) and empty addresses (23%, 6.1× lift).

**Per-country** (US and India both have full labels): US **0.9896**, India **0.9749**. India's
gap is blocking recall (95.70% vs 98.57%). France has **no labels anywhere** — it exists only in
the unlabelled test set, so it can never be scored directly. Fixing India is the tractable proxy.

### Dead ends — already measured, do not repeat

| Idea | Result |
|---|---|
| Better global threshold | Best single τ is **worse** than the shipped rule (−0.000274) |
| Per-country thresholds | **−0.000083**. No gain. |
| Segment-specific cuts (non-Latin / empty-address) | Model is already well calibrated there |
| Hand-tuned similarity rules replacing the GBM | GBM AUC is 0.99996; rules will lose |
| "Add a phone/domain feature" | **There is no phone field.** Columns are only `entity_id, business_name, business_address, country` |

`oracle_topk` = 0.990587 and `oracle_subset` = 0.990727 differ by 0.00014, so **ranking is
effectively solved**. The remaining decision-layer headroom (≈0.0065) is entirely in choosing
**k per entity**, which is a group-size *prediction* problem, not a threshold problem.

---

## Where to spend your time, in measured order

Loss splits as: **blocking 0.0093**, matcher+decision **0.0069**. See `LITERATURE.md` for which
published techniques were tested against our data and which failed to transfer.

1. **Union BM25 top-5 into blocking.** **Already measured** (`src/blocking_probe.py`):
   +0.003026 on the oracle ceiling for +1.24 candidates/entity, rescuing 29.5% of the
   incumbent's misses. The two blockers fail on different pairs. Implement the union in
   `blocking.py`, then measure *realized* `holdout_stage2_f05` — expect +0.002–0.003.
   ⚠️ **BM25 as a *replacement* was tested and is much worse** (94.29% @k=40 vs 97.42% @21).
   Union only.
2. **Turn on `--dense`. It was OFF in every run so far.** `multilingual-e5-small` (MIT, 118M)
   embeds Devanagari and Latin in one space, directly targeting the 37% of misses with
   non-Latin names. Already implemented — just pass the flag. Cheapest untested lever.
3. **Blocking retrieval** (63% of errors): mine a Devanagari→English token table from ground
   truth (`पावर`→`power`, `मीडिया`→`media`) — note `normalize.py` already emits `n_translit`,
   `n_phon`, `n_skel`, so check what blocking actually keys on before adding fields; make
   `max_df` caps adaptive so a record whose only shared tokens are common ones still retrieves
   something; give address-keys and name-keys separate K budgets (the failure modes are disjoint).
4. **Per-entity group-size prediction** (the 0.0065): predict n_S2 and n_S3 from the candidate
   score profile, clipped to the proven caps, then take top-n.
5. **Precision work — last.** It is 7.8% of errors.

Items 1–3 are all recall plays: **stack them, then measure together** against the +0.005 gate.

---

## Layout

```
amazon-ml-challenge-2026/
├── AGENTS.md          <- you are here
├── EXPERIMENTS.md     <- THE LOG. append to it every time.
├── FINDINGS.md        <- measured evidence; supersedes other docs
├── README.md          <- project index
├── research.md        strategy.md   PLAN.md   HARDWARE.md
├── code/business_entity_resolution/    <- THE PIPELINE (submission layout)
│   ├── src/           pipeline + analysis tools
│   ├── tests/         smoke_test.py
│   ├── .venv/         python 3.11 env (already built)
│   ├── work_dev/      dev-scale cache (disposable)
│   └── requirements.txt  requirements-dense.txt  package_submission.py
├── student_resource/  given data, official validator, Documentation_template.md
├── baseline/output_0.945193/   the scored 0.945 submission, archived
├── output/            live submission artifacts
└── work/              analysis scratch (sample.json, holdout ids)
```

## Setup and verify (2 minutes)

```bash
cd code/business_entity_resolution
# the venv already exists; rebuild only if needed:
#   uv venv .venv -p 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/smoke_test.py        # must print SMOKE TEST PASS
```

## The standard measurement loop

```bash
cd code/business_entity_resolution
DS=../../student_resource/dataset

# 1. train + measure on the dev slice (~3 min). --keep-intermediates is REQUIRED
#    for the analysis scripts: the pipeline deletes stage-2 features otherwise.
.venv/bin/python src/run.py train --data-dir $DS --work-dir work_dev \
    --dev-frac 0.03 --keep-intermediates

# 2. the headline numbers
cat work_dev/models/report.json

# 3. where the errors are now
.venv/bin/python src/error_analysis.py   --work-dir work_dev
.venv/bin/python src/profile_misses.py   --work-dir work_dev
.venv/bin/python src/decision_analysis.py --work-dir work_dev
```

Compare `holdout_stage2_f05`, `holdout_blocking_recall`, `holdout_oracle_f05`,
`pairs_per_s1` and `per_country_holdout_f05` against the baseline row in `EXPERIMENTS.md`.
Then **log it**.

### Which number to trust

| Metric | Meaning |
|---|---|
| `holdout_blocking_recall` | fraction of true pairs that became candidates |
| `holdout_oracle_f05` | **hard ceiling** given the candidate set. If this is below your target, model work cannot save you — fix blocking. |
| `holdout_stage2_f05` | what the model actually scores |
| `pairs_per_s1` | candidate-set size — **explicitly graded**, keep it small |
| `per_country_holdout_f05` | the per-country split |

Dev-slice scores are **optimistic** (`_dev_subsample` warns: fewer name collisions). The slice
scores ~0.984 where full scale scored 0.945. **Trust deltas, never absolutes**, and always
confirm at `--dev-frac 0.3` before promoting.

## Analysis tools

| Script | Purpose |
|---|---|
| `src/diagnose.py` | official metric on any submission: per-country, blocking ceiling, FP-vs-FN headroom, cap violations, deterministic holdout. Stdlib+numpy, runs anywhere. Verified: GT vs GT = exactly 1.000000 |
| `src/error_analysis.py` | labels the trained model's holdout errors into the taxonomy |
| `src/profile_misses.py` | why blocking missed: trait lift among missed vs found pairs |
| `src/decision_analysis.py` | decision-layer bounds: best global τ, per-country τ, oracle top-k, oracle subset |
| `src/verify_gt.py` | verifies generator transformations; `--one-owner` proves the structural constraints |

## Hard constraints — violating these disqualifies the submission

- **No external data lookup.** No entity-resolution APIs, no government registries, no
  geocoding, no internet augmentation. Only the provided files. Top teams are audited.
  Static abbreviation/state dictionaries and IDF computed from the given data are fine; a
  table you copied from a website is not.
- **Final model MIT/Apache-2.0, ≤ 8B parameters.** LightGBM (MIT) and
  multilingual-e5-small (MIT, 118M) both qualify. Record whatever you use.
- **Validate before every upload:**
  ```bash
  python3 student_resource/utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir student_resource/dataset/test
  ```
  Must print `PASS`. A rejected upload wastes a submission slot.
- Every test S1 entity gets exactly one row; empty list for singletons; S2/S3 ids only; no
  duplicates; matches must be a subset of candidates.
- **Decide on the holdout, not the public leaderboard.** The private split sets the rank.

## Working habits

- **One change per run.** Two at once and the delta teaches you nothing.
- Keep every `report.json`. Copy it next to its `EXPERIMENTS.md` row if you change configs.
- The pipeline caches by stage; re-running skips finished work. `--force` recomputes.
- `--dev-frac 0.03` ≈ 3 min, `0.3` ≈ 30 min, full ≈ 4–9 h.
- If you are about to hand-write a dictionary of aliases, **mine it from ground truth
  instead**. The data is a synthetic generator with a finite rule set; the ground truth
  contains the rules. Hand-written lists have already misfired once here — "appended
  descriptor words" turned out to be 0.94% of pairs, not a main pattern.
- State clearly what you verified vs assumed. This project has already been burned twice by
  confident inference: a fabricated 0.995–0.998 ceiling, and "France is the dominant leak"
  when India (fully labelled) was measurably worse.
