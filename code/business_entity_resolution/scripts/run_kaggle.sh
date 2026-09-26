#!/usr/bin/env bash
# Kaggle Notebook GPU (T4/P100, ~16GB VRAM, ~30GB RAM) — full pipeline.
# Conservative batches/RAM vs A100. Stages cache under WORK for session resume.
#
# On Kaggle Notebook:
#   1. Settings → Accelerator → GPU
#   2. Add dataset input (or place TSVs under $DS)
#   3. bash scripts/run_kaggle.sh
#
# Local test of flags only — do not expect full data speed on Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
DS="${DS:-/kaggle/input/amazon-ml-ber-dataset}"
# Kaggle often mounts a single folder with train/ test/ inside:
if [[ ! -d "$DS/train" && -d /kaggle/input ]]; then
  for d in /kaggle/input/*; do
    if [[ -d "$d/train" && -d "$d/test" ]]; then DS="$d"; break; fi
    if [[ -d "$d/dataset/train" ]]; then DS="$d/dataset"; break; fi
  done
fi
WORK="${WORK:-/kaggle/working/work_full}"
OUT="${OUT:-/kaggle/working/output}"
PY="${PY:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export CUDA_MODULE_LOADING=LAZY
export KMP_DUPLICATE_LIB_OK=TRUE
# Keep Kaggle from OOM-killing the session
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

mkdir -p "$WORK" "$OUT"

echo "=== device ==="
"$PY" - <<'PY'
import torch
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram_free_gb={free/2**30:.1f} total_gb={total/2**30:.1f}")
PY
echo "DS=$DS"
test -d "$DS/train" || { echo "ERROR: $DS/train missing — attach the competition dataset"; exit 1; }

# T4/P100 16GB: batch 512 safe; 1024 if free VRAM > 10GB after model load
BATCH="${DENSE_BATCH:-512}"
WORKERS="${WORKERS:-4}"

echo "=== full all (Kaggle-safe) ==="
exec "$PY" -u src/run.py all \
  --data-dir "$DS" \
  --work-dir "$WORK" \
  --out-dir "$OUT" \
  --dense \
  --dense-batch "$BATCH" \
  --bm25-k 5 \
  --bm25-k-empty 40 \
  --k-dense 15 \
  --k-dense-script 50 \
  --feat-fuzz-brank-max 24 \
  --feat-chunk 1500000 \
  --join-budget-rows 25000000 \
  --workers "$WORKERS" \
  --max-train-rows 30000000
