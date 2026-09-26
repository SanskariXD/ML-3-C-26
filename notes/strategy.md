# Strategy — Amazon ML Challenge 2026 (Business Entity Resolution)

> The decision and the reasoning. Measured facts in `research.md`; step-by-step execution in
> `PLAN.md`; machine setup in `HARDWARE.md`.

## Where we are

| | |
|---|---|
| Baseline | **0.945193** |
| Top-10 cut | **0.986** |
| Gap to close | **0.041** |
| Structural ceiling | **≈0.999** (measured — see `FINDINGS.md`) |
| Target | **0.999 is defensible**; commit ≥0.986, stretch 0.995 |

> **Superseded by `FINDINGS.md`.** An earlier version of this file claimed a 0.995–0.998
> ceiling and named France as the dominant leak. Both were wrong. The ceiling rested on
> "14% of matches have no name signal", which was unnormalized token mismatch, not missing
> information — the true figure is **2.36%**, so the ceiling is ~0.999. And measured
> per-country scores show **India (0.9749) trails US (0.9896) despite having full labels**,
> so the problem is non-US *structure*, not only France's missing labels.

## Thesis

Measured on the real pipeline against real ground truth: **92.2% of all errors are false
negatives, and 63.1% of all errors are blocking misses** — true matches that never entered the
candidate set. Precision errors total 7.8%.

The classifier is already near-perfect at ranking (stage-1 AUC 0.99994, stage-2 AUC 0.99996),
which is exactly why the tuned decision rule is so conservative that it over-merges almost
nothing. **The score is starved of recall.** F₀.₅ being precision-weighted misled everyone —
including the baseline authors and my own first plan — into defending precision that was never
under threat.

And the misses are not irreducible: **99.92% of missed pairs share at least one token with
their S1 record.** Blocking had the signal and failed to retrieve it, consistent with the
`max_df_pair=500` / `max_df_name=200` caps and the top-K=40 cut discarding pairs whose only
shared tokens are common ones. Two nearly disjoint archetypes:

- **Name unusable (72% of misses)** → retrievable only by address, which does share tokens.
- **Address empty (23% of misses)** → retrievable only by name, under the stricter df cap.

Devanagari is the concentration point: 37% of misses have non-Latin names (6.1× lift), and
`anyascii` transliterates phonetically, so `पावर` → `pavara`, which never token-matches `power`.

## The levers, in measured expected-value order

Loss decomposes as: **blocking ceiling 0.0093**, matcher+decision **0.0069**.

### 0. Turn on `--dense` — do this first, it is already built
The dense pass was **off** in the diagnostic run. `multilingual-e5-small` (MIT, 118M) embeds
Devanagari and Latin into one space, which directly attacks the 37% of misses with non-Latin
names — the largest single identified cause. Zero new code: `run.py all --dense`. Cheapest
high-value experiment available.

### 0b. Fix blocking retrieval — 63% of all errors
- **Mine a Devanagari→English token table from ground truth** (पावर→power, मीडिया→media,
  प्राइवेट→private, लिमिटेड→limited). Finite business/city vocabulary; fixes precisely what
  phonetic transliteration cannot.
- **Make `max_df` caps adaptive.** A record whose only shared tokens are common ones currently
  retrieves nothing. Raise the cap when a record has few usable keys.
- **Separate K budgets for address-keys and name-keys**, since the two failure archetypes are
  disjoint: empty-address records need name-only keys, non-Latin-name records need address-only.
- Sweep `k_key` and report ceiling vs candidates/entity, picking the knee.

### 0c. Re-tune the decision for recall — 29% of all errors
Precision is measurably cheap (7.8% of errors), so the conservative τ/γ is leaving score on the
table. Predicted mean is 3.218 vs truth 3.461. Re-tune explicitly against the measured error
mix rather than against the assumption that false merges dominate.

### 1. Supervise France synthetically — now a secondary lever
India trails US by 0.0147 *with* full labels, so fixing non-US structure (above) is the
tractable proxy and should transfer to France. Synthesis remains worthwhile but is no longer
the headline.
We know the generator's rule families from US+India ground truth. Mine their empirical
distribution (suffix-append rate, transposition rate, typo rate per character, abbreviation
substitutions, empty-address rate, number-perturbation rate, reorder rate, distractor rate),
then **apply those rules to French Source-1 records to synthesize labelled French pairs** —
including realistic distractors so the negative distribution stays honest. Train on
US + India + synthetic-France.

Learn French alias tables **unsupervised from the provided files**: cluster admin labels that
co-occur with the same city token to recover region↔department pairs; recover street-type
abbreviations from prefix/context statistics. This uses only supplied data, so it stays inside
the fair-play rules — no geocoding, no external registry.

*Validation:* leave-one-country-out. Train on US only, score India. The existing
`--train-countries US` flag makes this a single command, and the drop is a direct read on the
France penalty.

### 2. Raise the blocking ceiling while shrinking the candidate set
Both halves of this are rewarded: recall bounds the score, and candidate-set size is ranked.
Evidence dictates the design:
- Address is the backbone (non-empty addresses always share a token) → IDF-weighted address
  retrieval over **learned-alias-normalized** tokens.
- Number-anchored compound keys `(house_number, rare_street_token_prefix)` — 92% of pairs
  share a number, and these keys have tiny posting lists.
- Name-only keys to cover the 4.4% with empty addresses.
- Never require name similarity: ~14% of matches have none.

*Target:* ceiling **≥0.995** at **≤10 candidates/entity**, down from 21.46.

### 3. Decide collectively, not pairwise
- **Group-size head:** predict n_S2 and n_S3 for each entity from the candidate score profile,
  then take top-n. The count distribution is tight and capped, and this directly repairs the
  measured under-prediction (3.218 vs 3.461).
- **Competition features:** for each candidate, the margin between its score for this entity
  and its best score for any *other* entity — turning the one-owner constraint from a hard
  greedy mask into a learned signal.
- **Cross-source propagation:** if an S2 and an S3 record are near-duplicates of each other and
  one matches the entity strongly, the other probably does too. Group members all descend from
  the same S1 record, so this evidence is real.
- **Global assignment:** replace greedy exclusivity with constrained optimization (min-cost
  flow / auction) over connected components, maximising summed expected F₀.₅ subject to
  one-owner and the caps.

### 4. Cross-encoder rerank on the uncertain band
Score only the pairs where LightGBM is undecided (p ∈ [0.05, 0.95], roughly 10–15% of
candidates) with a small multilingual cross-encoder fine-tuned on hard pairs. This is where
the GPU earns its keep and where the last ~0.01 lives — it reads Devanagari and French
natively, which hand-built features cannot. Model must be MIT/Apache and ≤8B (e5-small 118M,
or bge-m3 568M — both comfortably legal).

### 5. Free precision
- **Enforce the caps.** 348 entities currently violate n_S2 ≤ 5 / n_S3 ≤ 6; those extras are
  guaranteed false positives. Costs nothing, worth ~0.0002.
- **Per-country probability calibration** (isotonic) so the expected-F₀.₅ arithmetic is valid
  for France, not just for the countries it was tuned on.
- **Per-country decision parameters** instead of one global τ/γ.

## Guiding principle

Measure before modelling. `code/business_entity_resolution/src/diagnose.py` reports the exact
metric, a per-country breakdown, the blocking ceiling, and — most importantly — the
**FP-vs-FN headroom split**, which says which direction to push. It is verified: ground truth
scored against itself returns exactly 1.000000. No model change ships without a gate number
from it.

## Scope

**Must have**
- [ ] Per-country F₀.₅ on a deterministic holdout (the number we never had)
- [ ] LOCO measurement of the unseen-country penalty
- [ ] Blocking ceiling ≥0.995 at ≤10 candidates/entity
- [ ] Synthetic France supervision
- [ ] Cap enforcement + per-country calibration
- [ ] Validator passing on every upload

**Stretch**
- [ ] Cross-encoder rerank on the uncertain band
- [ ] Global assignment via min-cost flow
- [ ] Group-size prediction head

**Explicitly cut**
- Any external data, API, geocoding or registry lookup — instant disqualification.
- Chasing 0.999. It is below the noise floor of the generator's ambiguity.
- Large LLM matchers: slower and worse than a calibrated GBM on 37M pairs.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| France hypothesis is wrong and loss is uniform | Medium | LOCO test costs one run and settles it before any build |
| Synthetic France pairs don't match the real generator | Medium | Validate by synthesizing *India* from US rules and checking the score transfers |
| Disk exhaustion mid-run (only ~131 GB free) | **High** | Budget in `HARDWARE.md`; `keep_intermediates=False`; free to 250 GB |
| Blocking shrink costs recall | Medium | Pick K at the knee of the measured recall curve, never blind |
| Overfitting the public leaderboard | Medium | Decide on the holdout; private split is what ranks |
