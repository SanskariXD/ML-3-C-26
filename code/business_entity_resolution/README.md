# Business Entity Resolution — how to run

Pipeline: **normalise → IDF multi-key blocking ∪ BM25 ∪ multilingual-e5 dense ∪ script
neighbors → LightGBM → isotonic → expected-F₀.₅ + housenum adjust + one-owner.**

Dense, BM25-k5, empty-addr BM25-k40, script neighbors, and housenum p-adjust are **ON by
default** (measured keeps — see `../../EXPERIMENTS.md`).

> **New here?** Read `../../AGENTS.md` first, then `../../FINDINGS.md`. Log every change in
> `../../EXPERIMENTS.md`.

---

## 1. Environment

Python **3.11**. A prebuilt `.venv/` may already be present.

```bash
# from this directory
uv venv .venv -p 3.11
uv pip install --python .venv/bin/python -r requirements.txt

# dense is default-ON — install the encoder deps
uv pip install --python .venv/bin/python -r requirements-dense.txt

# optional CUDA (RTX 50-series / Blackwell needs cu128, not cu121):
# uv pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Apple Silicon already has torch/MPS via the dense requirements. Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())"
```

## 2. Smoke test (~1–2 min)

```bash
.venv/bin/python tests/smoke_test.py          # must print: SMOKE TEST PASS
```

## 3. Dev-scale loop (live here)

Full runs take hours–days. Iterate on a slice.

```bash
DS=../../student_resource/dataset

# Mac 16–24 GB unified memory (default --dense-batch 256 is correct):
.venv/bin/python -u src/run.py train --data-dir $DS --work-dir work_dev \
    --dev-frac 0.03 --keep-intermediates --dense-batch 256

cat work_dev/models/report.json
```

**`--keep-intermediates` is required** for `error_analysis.py` / `profile_misses.py` /
`decision_analysis.py` — otherwise stage-2 features are deleted after training.

Dev-slice scores are **optimistic** (~0.99 where full scale was ~0.945). **Trust deltas,
not absolutes.** Confirm at `--dev-frac 0.3` before promoting.

```bash
.venv/bin/python src/error_analysis.py    --work-dir work_dev
.venv/bin/python src/profile_misses.py    --work-dir work_dev
.venv/bin/python src/decision_analysis.py --work-dir work_dev
```

## 4. RAM / batch sizes (read this before a long run)

Last night-style crashes are almost always **MPS/CUDA OOM** or **memory pressure from a
too-large encode batch**, not a logic bug. The encode path probes batch sizes unless you
pin with `--dense-batch` (that also sets `dense_batch_fixed=True` and skips the probe).

| Machine | RAM / VRAM | Recommended flag | Notes |
|---|---|---|---|
| Mac M-series, **16 GB** | unified | `--dense-batch 128` | If you see jetsam / process killed, drop to 64 |
| Mac M-series, **24 GB** (M5 Air etc.) | unified | `--dense-batch 256` | Default. 512+ has hung / OOM'd on MPS |
| Mac M-series, **36+ GB** | unified | `--dense-batch 384` | Probe can try 512; pin if unstable |
| NVIDIA, **16 GB VRAM** | discrete | `--dense-batch 512` | Start here; raise if free |
| NVIDIA, **24+ GB VRAM** | discrete | `--dense-batch 1024` | Sweet spot used in HARDWARE.md |
| NVIDIA, **40+ GB VRAM** | discrete | `--dense-batch 1536` | Only if `nvidia-smi` shows headroom |

### If it crashed / was killed

1. **Lower `--dense-batch`** one step (256 → 128 → 64) and re-run with the **same**
   `--work-dir` — finished stages are cached; only the failed stage recomputes.
2. Do **not** set `OMP_NUM_THREADS` yourself to a huge number before importing torch —
   `run.py` already caps threads. Mixing a high OMP with torch+LightGBM has deadlocked
   before; the entrypoint re-execs to avoid that.
3. For full runs leave **`--keep-intermediates` off** (default). Keeping every memmap can
   push disk past 60 GB and thrash RAM via page cache.
4. Use a **fresh `--work-dir` name per experiment** (`work_exp_foo`). Delete old ones:
   `rm -rf work_*` — they are gitignored and regenerable.
5. Prefer `python -u` (unbuffered) so a kill still leaves the last log line on disk under
   `work_dir/run_*.log`.

### Disk budget (full scale, both splits)

| Mode | Peak |
|---|---|
| Default (`keep_intermediates=False`) + dense | ~40 GB |
| Everything retained + dense | ~60 GB |

## 5. Full run (only after the `AGENTS.md` promotion gate)

```bash
# Mac:
.venv/bin/python -u src/run.py all --data-dir $DS --work-dir work_full \
    --out-dir ../../output --dense-batch 256

# CUDA box:
.venv/bin/python -u src/run.py all --data-dir $DS --work-dir work_full \
    --out-dir ../../output --dense-batch 1024
```

Produces `../../output/matching_results.tsv` and `../../output/candidate_pairs.tsv`, then
self-check + official validator.

| command | what it does |
|---|---|
| `run.py prepare --split train` | normalise → parquet |
| `run.py block --split train` | candidates (also prepare) |
| `run.py blocking-report` | recall ceiling, pairs/S1 |
| `run.py features --split train` | feature memmaps |
| `run.py train` | fit + `work/models/report.json` |
| `run.py predict` | test inference + outputs |
| `run.py all` | train then predict |

Useful flags: `--dev-frac 0.1`, `--train-countries US`, `--k-key 60`,
`--bm25-k 5`, `--bm25-k-empty 40`, `--k-dense 15`, `--k-dense-script 50`,
`--no-dense` (ablate), `--force`, `--keep-intermediates`.

## 6. Which metric to trust

| metric | meaning |
|---|---|
| `holdout_blocking_recall` | true pairs that became candidates |
| `holdout_oracle_f05` | **hard ceiling** given candidates — if low, fix blocking |
| `holdout_stage2_f05` | what the model scores |
| `pairs_per_s1` | candidate-set size (graded — keep small) |
| `per_country_holdout_f05` | US vs India (France has no labels) |

## 7. Layout

```
src/
  run.py        CLI
  config.py     every knob + feature contract
  blocking.py   IDF keys ∪ BM25 ∪ empty-addr keep
  dense.py      e5 embeddings + kNN (batch probe / pin)
  features.py   rapidfuzz + TF-IDF + group context
  pipeline.py   LightGBM CV, calibration, inference
  decide.py     one-owner, expected-F, housenum adjust
  diagnose.py error_analysis.py profile_misses.py decision_analysis.py verify_gt.py
tests/smoke_test.py
package_submission.py
```

## 8. Package + validate

```bash
.venv/bin/python package_submission.py --team MyTeam

python3 ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```

Must print `PASS`.

## 9. Licences and fair play

LightGBM (MIT) + multilingual-e5-small (MIT, 118M). No external data lookup — static
abbreviation dictionaries and IDF from the given files only. Top teams are audited.
