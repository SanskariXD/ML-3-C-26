# Research — Amazon ML Challenge 2026 (Business Entity Resolution)

> Measured facts only, all reproduced from `student_resource/dataset` on 2026-09-26.
> Decisions live in `strategy.md`; the execution plan in `PLAN.md`.
> **Pure ML / data-science hackathon — no blockchain.**

## Hackathon

| | |
|---|---|
| Name | Amazon ML Challenge 2026 |
| URL | https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604 |
| Organizer | Amazon, hosted on Unstop |
| Problem | Business Entity Resolution across 3 noisy sources with no shared identifiers |
| Metric | **F₀.₅**, macro-averaged per Source-1 entity (precision-weighted 2×) |
| Our baseline | **0.945193** |
| Top-10 cut | **0.986** |
| Structural ceiling | **≈0.999** (measured — the ≈0.995–0.998 figure derived below is **retracted**, see `FINDINGS.md`) |

### Task

For every Source-1 entity (the deduplicated reference source), find all matching records in
Source 2 and Source 3. An entity may match zero, one, or many records.

### Metric, exactly

```
G = |true|,  P = |predicted|,  TP = |true ∩ predicted|
G == 0 :  F = 1.0 if P == 0 else 0.0
G  > 0 :  F = 0.0 if P == 0 else 1.25 · TP / (0.25 · G + P)
```
Macro-averaged over **all** S1 entities, singletons included. Verified against the
problem statement's worked example in `code/business_entity_resolution/src/diagnose.py`
(returns 0.714286 ✓).

**Cost of one error at G=3:** one false positive −0.211, one false negative −0.091.
**A false positive is 2.3× more expensive than a miss.**

### Ranking also weighs blocking

Per the update on the code page, `candidate_pairs.tsv` counts toward final ranking: judges
review recall ceiling and reduction ratio, and **a smaller candidate set per S1 entity ranks
higher**. Our current set averages **21.46 candidates/entity** — a target for reduction.

### Hard rules

- Output must validate exactly (`utils/validate_submission.py`) or it is not scored.
- Final model: **MIT/Apache-2.0**, **≤ 8B parameters**.
- **No external data lookup** — no ER APIs, no government registries, no geocoding, no
  internet augmentation. Only the provided data. Top teams are audited.

---

## Measured dataset shape

| File | Rows | Size |
|---|---|---|
| `train_source1.tsv` | 2,206,821 | 200 MB |
| `train_source2.tsv` | 5,034,616 | 467 MB |
| `train_source3.tsv` | 5,285,603 | 480 MB |
| `train_ground_truth.tsv` | 2,206,821 | 121 MB |
| `test_source1.tsv` | 1,732,544 | 167 MB |
| `test_source2.tsv` | 4,887,273 | 486 MB |
| `test_source3.tsv` | 5,082,316 | 483 MB |

**24.23M source records** (12.53M train + 11.70M test), **2.3 GB** — ~26.4M counting
ground-truth rows. This is a scale problem, not a toy.

### Ground-truth distribution

- **Singletons: 5.585%** (123,247 entities)
- **Mean matches: 3.461**
- **Hard structural caps: n_S2 ≤ 5, n_S3 ≤ 6**, total ≤ 11 — across all 2.2M entities
- Top patterns (n_S2, n_S3): (1,1) 12.2%, (1,2) 11.4%, (2,1) 10.1%, (2,2) 9.4%
- **Distractors:** ~1.34M S2 and ~1.34M S3 records (~27% of each) match no S1 entity

### Country composition (test Source-1)

| Country | Entities | Share | Training labels |
|---|---|---|---|
| India | 809,986 | 46.8% | yes |
| US | 663,106 | 38.3% | yes |
| **France** | **259,452** | **14.98%** | **none** |

France is ~15% of the scored set with **zero** supervision. This is the prime suspect for the
0.055 gap (see `strategy.md`).

---

## The generator is synthetic and learnable

Reading matched groups side by side reveals a finite rule set. This is the single most
important finding: the noise is **not** arbitrary, so it can be reversed.

**Source 2 transformations:** address UPPERCASED; street-type abbreviation
(Road→RD, Trail→TRL, Cove→CV, Avenue→AVE); appended name suffix
(Service / Center / Services / Co / Ltd / Enterprises); legal-suffix drop; word transposition
("Inc Silver Massage"); character typos ("Matsnise"); Devanagari transliteration; diacritic
injection (Únion); domain-form names (`qualityinsuranceunion.com`); bracket noise
(`[Products]`); **empty address**; `#` / `PO BOX` prefixes.

**Source 3 transformations:** address component **reordering**; state expansion/abbreviation
(NY↔New York, MH/KL/KA/UP); **city aliases** (Bangalore↔Bengaluru, Pune↔Poona,
X↔X Township, Mumbai↔Mumbai City); literal `<NULL>` tokens; **house-number perturbation**;
**completely random unrelated names** ("Dréxkor", "Belocalo") matched by address alone; name
truncation.

### Overlap statistics (20,063 sampled groups, 69,609 matched pairs)

| | S2 | S3 |
|---|---|---|
| empty address | 4.53% | 4.34% |
| non-Latin name | 8.77% | 4.65% |
| **name zero-overlap** | **16.0%** | **13.0%** |
| **address zero-overlap** | **0.00%** | **0.02%** |
| name exact token-set | 32.3% | 32.6% |
| address numbers shared | 91.5% | 92.3% |

**Two load-bearing conclusions:**

1. **A non-empty address always shares at least one token with its S1 address.** Address is
   the backbone of retrieval — an IDF-weighted address index should approach ~100% recall.
2. **~14% of true matches share no name token at all.** Any blocking that requires name
   similarity structurally loses them. Conversely the ~4.4% with empty addresses can only be
   found by name. **The union of both key families is mandatory.**

### France specifics (test only, no labels)

- Legal suffixes: SARL, SAS, SASU, EURL, SCI, S.A., S.A.S, Cie, Frères, Fils
- Street types: Rue / Boulevard / Avenue / Allée, abbreviated R., R, AV, Bd
- **Region ↔ department substitution** — the French analogue of the state-code swap:
  Dunkerque `Hauts-de-France` ↔ `Nord`; La Teste-de-Buch `Nouvelle-Aquitaine` ↔ `Gironde`
- Diacritic injection (Àmicale, ÀRT), `<<` and `(41)` bracket noise

These alias families can be learned **unsupervised from the provided test files** (city token
co-occurrence), which stays inside the fair-play rules.

---

## The ceiling: how good can anyone get?

| Measurement | Value |
|---|---|
| S1 records sharing an identical normalized **name + address** | **0 (0.0000%)** |
| S1 records sharing an identical normalized **address** | 104,295 (4.726%) |

Every S1 entity is **uniquely identifiable** by name+address — there is no truly unresolvable
collision. Ambiguity is confined to the intersection of "address collides with another S1"
(4.7%) and "name carries no signal" (~14%), i.e. **~0.7% of pairs**. At ~0.2 F₀.₅ cost each
that is ~0.0014 of irreducible loss.

> **⚠️ This derivation is RETRACTED — see `FINDINGS.md`.** The "~14% have no name signal" input
> conflated *unnormalized token mismatch* with *missing information*. Measured properly, only
> **2.36%** of true pairs have genuinely no name signal (the rest are domain forms, typos and
> transliterations, all recoverable). Revised ambiguity ≈ 0.11% of pairs ≈ 0.0008 loss, so the
> **ceiling is ≈0.999 and the 0.999 target is defensible.**

~~**Structural ceiling ≈ 0.995–0.998.** The requested 0.999 sits above it.~~ The objective is to
clear **0.986** (top 10) and push toward **0.995+**.

---

## Baseline under diagnosis (0.945193)

Prior pipeline (`README/`): normalize → IDF-weighted multi-key blocking (`k_key=40`) +
optional GPU dense pass (multilingual-e5-small, MIT, 118M) → 69 features → 2-stage LightGBM
(10 folds) → expected-F₀.₅ decision with a greedy one-owner constraint. Already sophisticated,
so the gap is **not** "add more features".

Measured defects in its output:

| Symptom | Evidence | Reading |
|---|---|---|
| **Under-prediction** | mean 3.218 ids vs truth 3.461; over-weights n=1–3, under-weights n=4–8 | ranking isn't confident enough, so the tuned rule stays conservative |
| **Cap violations** | 348 entities with n_S2=6,7 or n_S3=7,8 (0.0201%) | guaranteed false positives; free to remove |
| **Large candidate set** | 21.46 candidates/entity | directly penalised in final ranking |
| **No per-country metrics** | no `report.json` retained | the decisive diagnostic was never captured |

Sources retrieved 2026-09-26: problem statement `student_resource/README.md`; community dataset
mirror https://huggingface.co/datasets/logicalguy/amazon-ml-challenge-2026
