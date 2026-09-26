# Execution Plan — 0.945 → 0.986+

> ⚠️ **Phase order superseded by `FINDINGS.md` (measured 2026-09-26).** Phase 0's gates have
> been run; the results reorder everything below. Blocking is now first, France is demoted, and
> precision work is last.
>
> **Measured error mix:** blocking_miss **63.1%**, missed_in_candidates **29.1%**,
> fp_distractor 6.1%, fp_stolen 1.0%, fp_on_singleton 0.7% — i.e. **92.2% false negatives**.
> **Do these first, in this order:**
> 1. `run.py all --dense` — the dense pass was OFF; it targets the 37% non-Latin misses. Free.
> 2. Mined Devanagari→English token table + adaptive `max_df` caps + split K budgets.
> 3. Re-tune the decision for **recall** (precision is measurably cheap).
> 4. Then, and only then, France synthesis (§1) and precision work (§3.5).
>
> **Ceiling retracted:** ~0.999, not 0.995–0.998. Phase 0's LOCO step is superseded by the
> real per-country measurement: **US 0.9896 / India 0.9749**, India's gap being blocking recall
> (95.70% vs 98.57%).

> Evidence in `research.md`, reasoning in `strategy.md`, machine setup in `HARDWARE.md`.
> Every phase ends in a **gate**: a number from `diagnose.py`. No phase starts before the
> previous gate is green. That is what makes this bulletproof — not the volume of code, but
> the refusal to build on an unmeasured assumption.

**Committed target: ≥0.986. Stretch: 0.995. Measured ceiling: ~0.999 — 0.999 is defensible
(see `FINDINGS.md`).**

---

## Phase 0 — Instrument (no modelling) · ~2 h

You cannot fix what you cannot see. The baseline run kept no per-country metrics, so the most
important number in this project does not exist yet.

```bash
cd amazon-ml-challenge-2026
D=code/business_entity_resolution/src/diagnose.py
DS=student_resource/dataset

# deterministic 10% entity-level holdout (same ids on every machine)
python3 $D holdout --gt $DS/train/train_ground_truth.tsv --frac 0.1 --out work/holdout_ids.txt

# structural sanity of the current submission (no GT needed)
python3 $D caps --pred baseline/output_0.945193/matching_results.tsv --gt $DS/train/train_ground_truth.tsv
```

Then retrain the baseline on the 90% and produce holdout predictions + candidates, and run:

```bash
python3 $D score    --pred work/holdout_pred.tsv --gt $DS/train/train_ground_truth.tsv \
                    --source1 $DS/train/train_source1.tsv
python3 $D blocking --cand work/holdout_cand.tsv --gt $DS/train/train_ground_truth.tsv \
                    --source1 $DS/train/train_source1.tsv
```

**Gate 0 — three numbers on the table:**
1. `MACRO F0.5 CEILING` from `blocking`. **If it is below 0.99, blocking is the binding
   constraint and Phase 2 outranks everything else.**
2. Per-country F₀.₅ from `score` (US vs India). A large spread means country generalisation is
   already broken *inside* the training distribution.
3. `headroom if 0 FP` vs `headroom if 0 FN`. Whichever is larger names the direction of every
   later decision-rule change.

### Phase 0b — the decisive LOCO test · ~1 h

```bash
# train on US only, score India: a direct read on the unseen-country penalty
python src/run.py train --train-countries US --dev-frac 0.2
python3 $D score --pred work/loco_pred.tsv --gt $DS/train/train_ground_truth.tsv \
                 --source1 $DS/train/train_source1.tsv
```

**Gate 0b:** in-domain (US) F₀.₅ minus out-of-domain (India) F₀.₅ = the France penalty
estimate. If the drop is ≥0.10, Phase 1 is confirmed as the top priority and is worth ~0.02–0.03
of final score. If the drop is <0.03, the France hypothesis is wrong — **skip to Phase 2** and
treat the loss as uniform.

---

## Phase 1 — Supervise France · ~6 h · expected +0.015 to +0.030

Conditional on Gate 0b.

### 1.1 Mine the generator (`src/mine_rules.py`)
For every ground-truth pair, align S1 against its S2/S3 partner and extract the empirical
distribution of:
- appended name suffixes (Service, Center, Services, Co, Ltd, Enterprises, …) and their rates
- legal-suffix drop rate; token transposition rate; per-character typo rate
- address abbreviation substitutions (observed, both directions)
- **city alias pairs** and **state/region alias pairs** (the highest-value output)
- empty-address rate, `<NULL>` injection rate, component-reorder rate
- house-number perturbation distribution
- diacritic-injection and transliteration rates
- domain-form name rate; random-name-replacement rate; distractor rate

Emit `work/rules.json`. **This file replaces the hand-written dictionaries in `normalize.py`.**

### 1.2 Learn French aliases unsupervised (`src/fr_aliases.py`)
From the provided test files only: group records by rare city token, collect the admin labels
co-occurring with each, and cluster to recover region↔department pairs (Hauts-de-France↔Nord,
Nouvelle-Aquitaine↔Gironde). Recover street-type abbreviations (Rue/R./R, Avenue/AV,
Boulevard/Bd, Allée) from token-context statistics. **No external lookup — provided data only.**

### 1.3 Synthesize French supervision (`src/synth.py`)
Apply the mined rules to French Source-1 test records to emit labelled
`(S1, S2′, S3′)` groups that mimic the real generator, including:
- the true group-size distribution, respecting n_S2 ≤ 5 / n_S3 ≤ 6
- ~27% distractor records that match nothing
- the 5.6% singleton rate

Train on US + India + synthetic-France.

**Gate 1 (correctness check before trusting it):** synthesize *India* from US-mined rules only,
train on US + synthetic-India, and score real India. If that recovers most of the LOCO drop,
the synthesis is faithful and France will behave. If it does not, the rules are wrong — fix
them before spending a run on France.

---

## Phase 2 — Blocking: raise the ceiling, shrink the set · ~5 h · expected +0.005 to +0.015

Both halves are rewarded: recall bounds the score, and candidate-set size is explicitly ranked.

### 2.1 Re-normalize with mined aliases
Feed `work/rules.json` into `normalize.py`. Every alias recovered here converts a previously
unreachable pair into a reachable one.

### 2.2 Key families (union — all three are load-bearing)
| Family | Purpose | Evidence |
|---|---|---|
| IDF-weighted address tokens | the backbone | non-empty addresses **always** share ≥1 token |
| `(house_number, rare_street_token[:4])` | high precision, tiny postings | 92% of pairs share a number |
| name-core token keys | the empty-address 4.4% | those pairs have no address at all |

Never gate a candidate on name similarity: ~14% of true matches have zero name overlap.

### 2.3 Choose K on the measured curve
Sweep `k_key` ∈ {8, 12, 16, 20, 30, 40} and plot ceiling vs candidates/entity. Pick the knee.

**Gate 2:** ceiling **≥0.995** at **≤10 candidates/entity** (from 21.46). Report reduction ratio
in the methodology document — this is graded.

---

## Phase 3 — Collective decisions · ~6 h · expected +0.010 to +0.020

The measured under-prediction (mean 3.218 vs 3.461, distribution shifted toward n=1–3) is a
ranking-confidence problem, so fix ranking and the counts follow.

### 3.1 Competition features
For each candidate: its score for this entity minus its best score for any other entity; its
rank among that entity's candidates; the gap to the entity's top candidate. This converts the
one-owner constraint from a hard greedy mask into a learned signal.

### 3.2 Cross-source propagation
Build S2↔S3 near-duplicate similarity within each entity's candidate set. Feature: agreement
between this candidate and the entity's other high-scoring candidates. Justified because all
members of a true group descend from the same S1 record.

### 3.3 Group-size head
A small model predicting n_S2 and n_S3 from the candidate score profile, clipped to the caps.
Blend with the expected-F₀.₅ rule. Directly attacks the count shortfall.

### 3.4 Global assignment
Replace greedy `exclusive_mask` with min-cost flow / auction over connected components,
maximising summed expected F₀.₅ subject to one-owner + caps.

### 3.5 Free precision
Enforce n_S2 ≤ 5 / n_S3 ≤ 6 (348 violating entities today). Calibrate probabilities
**per country** (isotonic) and tune τ/γ/miss **per country**.

**Gate 3:** holdout F₀.₅ ≥0.985, predicted mean count within 0.05 of 3.461, zero cap
violations.

---

## Phase 4 — Cross-encoder rerank · ~4 h · expected +0.005 to +0.010

Only for pairs where the GBM is undecided (p ∈ [0.05, 0.95], ~10–15% of candidates). Fine-tune
a small multilingual cross-encoder on hard pairs — it reads Devanagari and French natively,
which no hand-built feature does. **License gate: MIT/Apache, ≤8B** (e5-small 118M or bge-m3
568M both qualify; record the choice in the methodology doc).

**Gate 4:** holdout F₀.₅ improves ≥0.002 over Phase 3, or the phase is dropped. Reranking is
the easiest place to overfit — trust only the holdout.

---

## Phase 5 — Ship · ~3 h

1. Full-scale run, both splits.
2. **Mandatory:** `python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test` → must print PASS.
3. `diagnose caps` on the final output → zero violations, count distribution matching truth.
4. Fill `student_resource/Documentation_template.md`: methodology, **blocking strategy with
   recall ceiling and reduction ratio**, architecture, features, and an explicit statement that
   no external data was used and the model is MIT/Apache ≤8B.
5. `python README/package_submission.py` → `<team>_submission.zip` in the required layout.

**Gate 5:** validator PASS, zero cap violations, candidates/entity ≤10, methodology complete.

---

## Sequencing and expected trajectory

| Phase | Effort | Cumulative | Confidence |
|---|---|---|---|
| 0 Instrument | 3 h | 0.945 (measured, not improved) | — |
| 1 France | 6 h | ~0.960–0.975 | medium-high |
| 2 Blocking | 5 h | ~0.970–0.985 | high |
| 3 Collective | 6 h | ~0.982–0.990 | medium-high |
| 4 Rerank | 4 h | ~0.985–0.992 | medium |
| 5 Ship | 3 h | final | — |

**If time is short, do 0 → 1 → 2 → 3.5 (caps + per-country calibration).** Those are the
highest ratio of score to effort. Phase 4 is genuinely optional.

## Rules of engagement

1. **Never submit without the validator passing.** A rejected upload wastes a submission.
2. **Decide on the holdout, not the public leaderboard.** The private split determines rank.
3. **One change per run.** Two simultaneous changes and you learn nothing from the delta.
4. **Iterate at `--dev-frac 0.1`** (10–20 min cycles); full runs overnight.
5. **Keep every `report.json`.** Losing per-country metrics is exactly why the baseline
   plateaued blind.
6. **No external data, ever.** Disqualification risk dwarfs any score gain.
