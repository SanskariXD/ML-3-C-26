#!/usr/bin/env bash
# Upload dataset + push/run the Kaggle notebook kernel (requires ~/.kaggle/kaggle.json).
#   1. https://www.kaggle.com/settings → Create New Token → save as ~/.kaggle/kaggle.json
#   2. Verify phone on Kaggle (needed for GPU)
#   3. bash kaggle/push_and_run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
command -v kaggle >/dev/null || pip3 install --user kaggle
test -f "$HOME/.kaggle/kaggle.json" || {
  echo "Missing ~/.kaggle/kaggle.json — download API token from kaggle.com/settings"
  exit 1
}
chmod 600 "$HOME/.kaggle/kaggle.json"
USER=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
echo "kaggle user=$USER"

# --- dataset (skip if already exists) ---
DS_SLUG="${USER}/amazon-ml-ber-dataset"
STAGE="$ROOT/kaggle/dataset_stage"
if ! kaggle datasets list -m 2>/dev/null | grep -q amazon-ml-ber-dataset; then
  echo "Creating dataset staging copy (2.3G)..."
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  rsync -a "$ROOT/student_resource/dataset/" "$STAGE/dataset/"
  cat > "$STAGE/dataset-metadata.json" <<EOF
{
  "title": "amazon-ml-ber-dataset",
  "id": "${DS_SLUG}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
  kaggle datasets create -p "$STAGE" --dir-mode zip
else
  echo "Dataset ${DS_SLUG} already listed — skipping create"
fi

# --- kernel ---
META="$ROOT/kaggle/kernel-metadata.json"
python3 - <<PY
import json
p="$META"
d=json.load(open(p))
d["id"]="$USER/ber-full-pipeline-gpu"
d["dataset_sources"]=["$USER/amazon-ml-ber-dataset"]
json.dump(d, open(p,"w"), indent=2)
print("kernel id", d["id"])
PY

kaggle kernels push -p "$ROOT/kaggle"
echo "Pushed. Starting status poll..."
for i in $(seq 1 120); do
  kaggle kernels status "$USER/ber-full-pipeline-gpu" || true
  sleep 60
done
