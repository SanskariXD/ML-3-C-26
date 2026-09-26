# Findings — ground-truth verification of the three checks

> Run 2026-09-26. Every number here is measured, reproducible, and produced by scripts in
> the repo. This file supersedes earlier assumptions in `strategy.md` and `PLAN.md`;
> where they conflict, this wins.

**Run configuration:** the real baseline pipeline, `run.py train --dev-frac 0.03`
(entity-consistent 3% sub-world, 13,378 holdout entities). Reproduced F₀.₅ = 0.984071
against the pipeline's own report of 0.983787 — the reconstruction is faithful.

⚠️ **Scale caveat, stated up front.** `_dev_subsample` warns its scores are optimistic
("fewer name collisions"). This slice scores 0.984 where the full-scale submission scored
0.945. So *absolute* numbers here are inflated; the *error mix* is the finding, and the
precision share is understated because a denser pool creates more confusable records.
Re-run at `--dev-frac 0.3` before betting the final architecture on the exact ratios.

---

## Check 1 — are the claimed transformations real?

Measured over 139,271 true pairs (`src/verify_gt.py`).

| Transformation | ALL | S2 | S3 | Verdict |
|---|---|---|---|---|
| street_type_abbreviated | 29.11% | 29.43% | 28.82% | confirmed (both sources, not S2-only) |
| component_reordered_partial | 27.97% | 27.49% | 28.43% | confirmed |
| **address_UPPERCASED** | 25.91% | **53.35%** | **0.01%** | confirmed — **a near-perfect S2/S3 discriminator** |
| number_perturbed | 15.67% | 16.69% | 14.72% | confirmed |
| name_truncated | 15.63% | 15.57% | 15.69% | confirmed |
| diacritic_injected | 13.62% | 15.55% | 11.79% | confirmed |
| legal_suffix_dropped | 13.51% | 14.12% | 12.94% | confirmed |
| address_component_dropped | 11.75% | 18.76% | 5.13% | confirmed |
| token_appended | 11.65% | 10.01% | 13.19% | confirmed |
| number_fully_changed | 7.94% | 8.35% | 7.56% | confirmed |
| non_latin_name | 6.67% | 8.94% | 4.52% | confirmed |
| domain_or_handle_name | 5.67% | 5.68% | 5.66% | confirmed |
| word_transposition | 5.55% | 5.70% | 5.40% | confirmed |
| typo_in_name | 5.50% | 5.55% | 5.44% | confirmed |
| address_EMPTY | 4.49% | 4.59% | 4.39% | confirmed |
| component_reordered (exact) | 4.05% | 6.43% | 1.79% | confirmed — **S3-skewed** |
| name_RANDOM_replacement | 2.08% | 1.73% | 2.42% | confirmed but **far rarer than claimed** |
| token_appended_descriptor | 0.94% | 0.78% | 1.09% | **overstated earlier** |
| null_literal_token | 0.78% | 0.85% | 0.72% | real but marginal |

**Corrections to the earlier write-up**
- "Appended descriptor words (Service/Center/Enterprises)" is only **0.94%**. Token appending
  is common (11.65%) but the appended token is usually *not* from that small vocabulary.
  Hard-coding that list would have been near-useless.
- `address_UPPERCASED` at 53.35% of S2 vs 0.01% of S3 is a strong, previously unnoticed
  **source-identity** signal.
- `<NULL>` literals are rare (0.78%) — not worth special handling.

### Structural properties — proven on the FULL training set

```
entities with >=1 match  : 2,083,574
distinct S2/S3 matched   : 7,638,365
records claimed by >1 S1 : 0   -> ONE-OWNER HOLDS
cap violations (S2>5|S3>6): 0   -> CAPS HOLD
```

Both are exact across 7.6M records. Safe to exploit as hard constraints.

---

## The ceiling claim was wrong — retracted

I previously asserted ~0.995–0.998 by multiplying "14% of matches have no name signal" by
"4.7% address collisions". That 14% was **unnormalized token mismatch**, not missing
information. Decomposed properly (139,271 true pairs):

| Category | Share |
|---|---|
| Name tokens already overlap | 85.63% |
| Recoverable: domain/concatenation form (`Bay Pediatrics` → `Baypediatrics.Com`) | 1.83% |
| Recoverable: trigram similarity — typo/truncation | 3.52% |
| Recoverable in principle: transliteration | 6.66% |
| **Genuinely no name signal** (`Maure Williams Colombier Inc` → `Dréxkor`) | **2.36%** |

2.36%, not 14% — and even that is overstated: the detector missed
`Maa Media Private Limited → maamedia.com` (TLD not stripped) and
`Barb Moon, DDS Center → ddsmoonbarb.com` (token-permuted domain).

**Revised: ~2.36% × 4.73% address-collision ≈ 0.11% of pairs genuinely ambiguous ≈ 0.0008
loss → ceiling ≈ 0.999.** The 0.999 target is defensible. The earlier ceiling is withdrawn.

---

## Check 2 — where the score actually leaks

Real model, real holdout, real ground truth (`src/error_analysis.py`). 13,378 entities,
1,627 (12.16%) with ≥1 error, 1,882 error items.

| Category | Count | Share |
|---|---|---|
| **blocking_miss** — true match never became a candidate | 1,188 | **63.12%** |
| **missed_in_candidates** — was a candidate, not selected | 548 | **29.12%** |
| fp_distractor — matched a record owned by nobody | 114 | 6.06% |
| fp_stolen — matched a record owned by another S1 | 19 | 1.01% |
| fp_on_singleton — matched anything on a true singleton | 13 | 0.69% |

### The headline

**92.2% of all errors are false negatives. Only 7.8% are precision errors.**
Of the recall loss, **68.4% is blocking's fault** and 31.6% is the model/decision's.

This inverts the intuition that F₀.₅ being precision-heavy means precision is where the work
is. The classifier is already near-perfect at ranking (**stage-1 AUC 0.99994, stage-2 AUC
0.99996**); the decision rule is consequently tuned very conservatively, so almost nothing is
over-merged. **The score is starved of recall, and blocking is the main culprit.**

### Why blocking misses (`src/profile_misses.py`, 1,194 missed pairs)

| Trait | MISSED | FOUND | Lift |
|---|---|---|---|
| no shared **name** token | 72.11% | 13.02% | 5.5× |
| **non-Latin name** (Devanagari) | 37.10% | 6.13% | 6.1× |
| **empty address** | 23.03% | 3.80% | 6.1× |
| address numbers disjoint | 14.15% | 7.28% | 1.9× |
| no shared address token | 0.08% | 0.01% | 9.4× |
| **no shared token at all** | **0.08%** | 0.00% | ∞ |

**The decisive number: 99.92% of missed pairs share at least one token with their S1
record.** Almost nothing was unreachable. Blocking had the signal and failed to retrieve it —
consistent with the `max_df_pair=500` / `max_df_name=200` caps and the top-K=40 cut discarding
true pairs whose only shared tokens are common ones.

Two failure archetypes, and they are nearly disjoint:
1. **Name unusable (72%)** → must be retrieved by address. The address *does* share tokens
   (only 0.08% don't), so these are top-K/df-cap casualties.
2. **Address empty (23%)** → must be retrieved by name. Since the hard intersection is ~0.08%,
   these pairs *do* share a name token, but name-only keys carry the stricter `max_df_name=200`.

Representative misses — note both the script change *and* the mangled address:

```
S1 'Tirupati Power'  | 'West Delhi, Wz-2D, New Delhi, Nangli Zalib B-1 Janak Puri, Delhi'
-> S3 'तिरुपति पावर'  | 'Wz-2d, West Delhi, DL, New Delhi'

S1 'Apex Media Private Limited' | '304, 3Rd Floor, Plot No. 1 Rg Complex, Rohini Sector-8, ...'
-> S2 'एपेक्स मीडिया प्राइवेट लिमिटेड' | 'Delhi, NORTH WEST, NEW DELHI, DOOR NO 82 304'
```

`anyascii` transliterates phonetically, so `पावर` → `pavara`, which does **not** token-match
`power`. Phonetic transliteration alone cannot bridge this; a mined Devanagari→English token
table can.

---

## Check 3 — per-country

France **cannot** be scored against ground truth: it appears only in the unlabelled test set,
so no French labels exist anywhere. LOCO is the only available evidence. What is measurable:

| Country | holdout F₀.₅ | blocking recall | missed pairs |
|---|---|---|---|
| **US** | **0.9896** | **98.57%** | 398 / 27,832 |
| **India** | **0.9749** | **95.70%** | 796 / 18,494 |

Error mix by country:

| Country | blocking_miss | missed_in_cand | fp_distractor | fp_stolen | fp_singleton |
|---|---|---|---|---|---|
| India | **71.6%** | 22.8% | 4.3% | 0.7% | 0.5% |
| US | 51.0% | 38.1% | 8.5% | 1.4% | 0.9% |

**India trails US by 0.0147 despite having full training labels.** So the country problem is
*not* purely "France has no labels" — it is that **non-US name/address structure is harder and
the normalization does not cover it**. India's gap is almost entirely blocking recall
(95.70% vs 98.57%), concentrated in Devanagari.

This refines the earlier France hypothesis rather than confirming it: France will likely suffer
from *both* the structural gap (measurable, fixable now via India) and label absence. Fixing
India's blocking is the tractable proxy and should transfer.

---

## Revised priority order

The loss decomposes as: blocking ceiling 1 − 0.9907 = **0.0093**, matcher+decision
0.9907 − 0.9838 = **0.0069**.

1. **Blocking recall — biggest, most certain win.** 63% of errors, and 99.92% of the misses
   are reachable. Concretely:
   - **Turn on `--dense`.** It was **off** in this run. `multilingual-e5-small` embeds
     Devanagari and Latin in one space, directly targeting the 37% non-Latin misses. This is
     the single cheapest high-value experiment and it is already implemented.
   - **Mine a Devanagari→English token table from ground truth** (पावर→power, मीडिया→media,
     प्राइवेट→private). Finite vocabulary; fixes what phonetic transliteration cannot.
   - **Make the `max_df` caps adaptive** — a record whose only shared tokens are common ones
     currently gets nothing. Allow higher df when a record has few usable keys.
   - Separate K budgets for address-keys vs name-keys (empty-address records need name-only).
2. **Recall in the decision layer** (29% of errors). Re-tune τ/γ for recall now that we know
   precision is cheap; predicted mean is 3.218 vs truth 3.461.
3. **India/non-US normalization**, which doubles as the France fix.
4. Precision work (7.8% of errors) — **lowest priority**, contrary to the original plan.
   Revisit only after confirming the mix at larger scale.

## Reproduce

```bash
cd code/business_entity_resolution
uv venv .venv -p 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/run.py train --data-dir ../../student_resource/dataset \
    --work-dir work_dev --dev-frac 0.03 --keep-intermediates
.venv/bin/python src/error_analysis.py --work-dir work_dev --show 40 --out work_dev/errors.tsv
.venv/bin/python src/profile_misses.py --work-dir work_dev
.venv/bin/python src/decision_analysis.py --work-dir work_dev
cd ../.. && python3 code/business_entity_resolution/src/verify_gt.py --sample work/sample.json
python3 code/business_entity_resolution/src/verify_gt.py \
    --one-owner student_resource/dataset/train/train_ground_truth.tsv
```

`--keep-intermediates` is required for the error scripts: the pipeline deletes stage-2
feature matrices by default (`pipeline.py:263`).
