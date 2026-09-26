#!/usr/bin/env bash
# A100 80GB / ≥64GB RAM full pipeline — winning accuracy stack, CUDA-tuned throughput.
# Usage (from code/business_entity_resolution):
#   bash scripts/run_a100.sh
#   DS=/path/to/dataset OUT=../../output WORK=work_full bash scripts/run_a100.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DS="${DS:-../../student_resource/dataset}"
WORK="${WORK:-work_full}"
OUT="${OUT:-../../output}"
PY="${PY:-.venv/bin/python}"

# Single OpenMP domain — do NOT also export OPENBLAS/MKL to different values.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export CUDA_MODULE_LOADING=LAZY
# faiss + torch both ship libomp; allow co-load on Linux GPU boxes.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

echo "=== device check ==="
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
"$PY" -c "import faiss; print('faiss', faiss.__version__, 'gpus', faiss.get_num_gpus())" || true

echo "=== full all (caches under $WORK) ==="
exec "$PY" -u src/run.py all \
  --data-dir "$DS" \
  --work-dir "$WORK" \
  --out-dir "$OUT" \
  --dense \
  --dense-batch 2048 \
  --bm25-k 5 \
  --bm25-k-empty 40 \
  --k-dense 15 \
  --k-dense-script 50 \
  --feat-fuzz-brank-max 24 \
  --feat-chunk 5000000 \
  --join-budget-rows 80000000 \
  --workers 16 \
  --max-train-rows 30000000
# Note: --dense-adaptive OFF until measured on --dev-frac 0.03 (then 0.3).
# Hard-neg sampling is automatic when rows > max-train-rows.
# Stages cache: prepare / emb_*.npy / pairs_*.npz / feat_*.npy — crash-safe resume.
