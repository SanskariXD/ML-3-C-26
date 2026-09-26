# Hardware Runbook — RTX 5080 Laptop

Target machine for full-scale runs.

| | | Implication |
|---|---|---|
| CPU | Intel Core Ultra 9 275HX @ 2.70 GHz (Arrow Lake-HX, 8 P-cores + 16 E-cores, 24 threads) | strong parallel blocking/features; **E-cores can slow LightGBM — benchmark thread count** |
| RAM | 32 GB (31.4 usable) | **the binding constraint** — everything large must be memmapped, never loaded |
| GPU | RTX 5080 Laptop, 16 GB GDDR7 (Blackwell, **sm_120**) | **requires CUDA 12.8+**; fits full dense index per country partition |
| Disk | 823 / 954 GB used → **~131 GB free** | workable, but keep intermediates off; free to ~200 GB for comfort |

---

## 1. Environment (Windows 11)

**Pin Python 3.11.** A stale cache showed a 3.14 run had been attempted; LightGBM /
pyarrow wheels lag on 3.14 and will either fail to build or fall back to slow paths.

```powershell
winget install astral-sh.uv
cd code\business_entity_resolution
uv venv .venv -p 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### GPU wheels — Blackwell needs CUDA 12.8

The RTX 5080 is `sm_120`. **cu121 wheels will not run on it** (you get "no kernel image is
available for execution on the device"). Install the cu128 build:

```powershell
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements-dense.txt
```

Verify before committing to a long run:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Expect `... True NVIDIA GeForce RTX 5080 Laptop GPU (12, 0)`. If capability prints `(12, 0)`
but `is_available()` is False, the driver is older than the toolkit — update the NVIDIA driver.

---

## 2. Disk budget

~24.2 M source records (12.53 M train + 11.70 M test). Estimated peak footprint:

| Artifact | Size |
|---|---|
| dataset (given) | 2.3 GB |
| normalized parquet caches | 4–5 GB |
| candidate pairs (≈157 M pairs, both splits) | ~2 GB |
| stage-1 features, fp16 × 69 cols | ~21 GB |
| stage-2 features | ~6 GB |
| dense embeddings, fp16 × 384-d | ~19 GB |
| models / folds / outputs | ~4 GB |
| **peak, `keep_intermediates=False`** | **~40 GB** |
| **peak, everything retained + dense** | **~60 GB** |

131 GB free is enough — with two conditions:

1. Leave `keep_intermediates: bool = False` (the default in
   `code/business_entity_resolution/src/config.py`). Its
   docstring already warns the retained matrices can exceed 40 GB on their own.
2. Put `work/` on the internal NVMe, not an external drive. Blocking and feature stages are
   IO-bound on random reads.

**Add a Defender exclusion for `work/`.** Real-time scanning on millions of memmap writes is
a large, silent slowdown:

```powershell
Add-MpPreference -ExclusionPath "C:\path\to\amazon-ml-challenge-2026\work"
```

---

## 3. RAM guards (32 GB)

The full stage-1 feature matrix is 88 M rows × 69 cols. As fp32 in RAM that is 24 GB and will
OOM; as an fp16 memmap read in chunks it is fine. Settings to adjust in `src/config.py`:

| Knob | Default | Set to | Why |
|---|---|---|---|
| `join_budget_rows` | 40,000,000 | **20,000,000** | raw join rows per chunk; the main OOM risk |
| `feat_chunk` | 2,000,000 | 1,000,000 | smaller feature chunks |
| `max_train_rows` | 30,000,000 | 20,000,000 if LightGBM OOMs | 30 M × 69 fp32 ≈ 8.3 GB plus histograms |

Watch peak RSS on the first full stage; if it crosses ~26 GB, halve `join_budget_rows` again.

---

## 4. Thread tuning — measure, don't assume

Hybrid P/E-core CPUs frequently run LightGBM **faster on P-cores only**, because the scheduler
lets slow E-core threads stall each histogram sync barrier.

```powershell
# benchmark both on a small slice before the long run
$env:OMP_NUM_THREADS=8;  python src\run.py train --dev-frac 0.05
$env:OMP_NUM_THREADS=24; python src\run.py train --dev-frac 0.05
```

Use all 24 threads for normalization, blocking and rapidfuzz features (embarrassingly
parallel); use whichever won for LightGBM.

---

## 5. Expected runtimes

Rough, for full scale on this machine:

| Stage | Time |
|---|---|
| prepare / normalize (24.2 M records) | 10–20 min |
| blocking, both splits | 30–60 min |
| features (≈157 M pairs, rapidfuzz) | 60–120 min |
| LightGBM 2-stage CV | 60–120 min |
| dense embeddings, e5-small fp16 @ len 64 | 45–90 min |
| cross-encoder rerank, ~8 M uncertain pairs | 45–90 min |
| inference + decision + output | 20–40 min |
| **full run, no dense** | **~4–6 h** |
| **full run, with dense + rerank** | **~6–9 h** |

Iterate with `--dev-frac 0.1` for 10–20 minute cycles. Reserve full runs for overnight.

---

## 6. VRAM plan (16 GB)

| Use | Footprint | Fits |
|---|---|---|
| e5-small (118 M) fp16 inference, batch 512 × len 64 | <2 GB | easily |
| exact GPU kNN index, 384-d fp16, 11.7 M records | ~9 GB | yes, per country partition |
| bge-m3 (568 M) cross-encoder fp16, batch 128 × len 128 | ~3 GB | yes |

Do kNN **per country partition** rather than globally — it keeps the index under 16 GB and is
also semantically correct, since matches never cross countries. Chunk the query side.

---

## 7. Overnight run checklist

- [ ] Plugged in; Windows power plan **Best Performance**
- [ ] Sleep and hibernate disabled (`powercfg /change standby-timeout-ac 0`)
- [ ] Laptop on a cooling stand, rear vents clear — sustained 24-core + GPU load will throttle
      a laptop chassis, and a throttled run is a slow run, not a failed one
- [ ] ≥100 GB free confirmed after `work/` is seeded
- [ ] Defender exclusion added for `work/`
- [ ] `torch.cuda.is_available()` verified True
- [ ] Smoke test passes: `python tests\smoke_test.py` → `SMOKE TEST PASS`
- [ ] Stage caching confirmed (re-running skips finished stages; `--force` recomputes)
- [ ] **`report.json` retained** — losing per-country metrics is why the baseline plateaued

---

## 8. macOS note

This workspace is on macOS, where the dense/GPU path is unavailable (no CUDA). Use the Mac for
data analysis, diagnostics (`src/diagnose.py`) and code edits; run blocking, training and the
dense passes on the 5080 machine, JarvisLabs (§9), or Vast.ai (§10). The diagnostics script is
stdlib + numpy only and runs anywhere.

---

## 9. JarvisLabs GPU VM — paid full-run compute

Source of truth: [GPU VMs (root SSH)](https://jarvislabs.ai/products/vm) ·
[India pricing](https://jarvislabs.ai/in) · [USD pricing](https://jarvislabs.ai/pricing).

Use this when the 5080 laptop is unavailable / RAM-bound and a change has already cleared the
dev-slice promotion gate in `AGENTS.md`. Do **not** burn a paid hour on a hunch.

### Why this provider (for us)

| Need | JarvisLabs fit |
|---|---|
| ≥64 GB system RAM (our OOM wall) | A30 = **112 GB**, L4 = **124 GB** |
| CUDA for `--dense` (e5-small) | A30 / L4 = **24 GB** VRAM (index ~9 GB) |
| CPU for blocking + rapidfuzz | L4 = **32 vCPU** (better); A30 = 16 vCPU |
| Budget ≤ ~₹500 / full run | A30 **₹38.88/hr**, L4 **₹41.31/hr** → ~6 h ≈ **₹230–250** |
| No AWS GPU quota | Instant; INR + GST invoice |

Prefer **On-Demand VM** (root SSH), not Templates: we need our own `uv` + `requirements.txt` +
cu128 torch, not a notebook image. Templates are fine only if you want Jupyter first.

### Pick one GPU (1×)

| GPU | VRAM | vCPU | RAM | India ₹/hr | USD $/hr | ~6 h cost | Use when |
|---|---|---|---|---|---|---|---|
| **L4** (recommended) | 24 GB | **32** | **124 GB** | ₹41.31 | $0.44 | ~₹250 | Full train + dense; more CPU for features |
| **A30** (cheapest) | 24 GB | 16 | 112 GB | ₹38.88 | $0.41 | ~₹235 | Same RAM class; slightly slower CPU stages |
| A100 40GB+ | skip | | | ≥₹84 | ≥$0.89 | ≥₹500 | Over budget for one overnight |

Skip H100 / H200 / RTX Pro 6000 — one hour alone can eat most of ₹500.

### Billing facts (from their docs)

- **Per-minute** GPU billing; no minimum rental length.
- **Pause** stops GPU charges; disk kept; storage still bills
  (~$0.00014/GB/hr on the VM page; also quoted as ~$0.10/GB·month on pricing).
- GPU is **dedicated** (no time-slicing). VM ≈ container GPU perf (±1.2% in their benches).
- Launch: VM ~**90 s** to root shell; Template ~1.8 s (prebuilt env).
- **$10 minimum credit** to start (USD page). That is ~₹850 once — above a ₹500 *run*
  budget, but leftover credit covers later runs. Check the India dashboard for INR top-up minima.
- India: pay by card (Stripe), **GST invoice**, regions India + Europe.

### Launch checklist

1. Sign up at [jarvislabs.ai](https://jarvislabs.ai) → add credit → pick **India** region if billed in ₹.
2. Dashboard: **VM** → **1× L4** (or A30) → disk ≥**150 GB** (dataset + work peak ~60 GB + headroom).
3. SSH in (`jl ssh <instance-id>` or the dashboard SSH line). Confirm:
   ```bash
   sudo whoami          # root
   nvidia-smi           # L4/A30 visible
   ```
4. Install toolchain (Python 3.11, not 3.14):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # clone or rsync the repo + student_resource/dataset
   cd amazon-ml-challenge-2026/code/business_entity_resolution
   uv venv .venv -p 3.11
   .venv/bin/uv pip install -r requirements.txt
   .venv/bin/uv pip install torch --index-url https://download.pytorch.org/whl/cu128
   .venv/bin/uv pip install -r requirements-dense.txt
   .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   .venv/bin/python tests/smoke_test.py   # must print SMOKE TEST PASS
   ```
5. Only after promotion gate: full train (expect ~4–7 h with dense):
   ```bash
   DS=../../student_resource/dataset
   .venv/bin/python src/run.py train --data-dir $DS --work-dir work --keep-intermediates
   # then: cat work/models/report.json  → copy into EXPERIMENTS.md
   ```
6. **Pause or destroy** the VM as soon as `report.json` + outputs are downloaded.
   Pausing keeps the disk for a cheap resume; destroy if you will not return.

### CLI shortcuts (from India docs)

```bash
jl setup                          # auth + optional agent skills
jl create --gpu L4 --vm           # root VM
jl ssh <instance-id>
jl run train.py --gpu L4          # one-shot upload+run (optional)
jl filesystem · jl vpc            # persistent dataset mount across instances
```

### What not to do

- Do not launch H100 “because it is faster” — wall clock is mostly CPU; you overpay.
- Do not leave the VM running overnight after the job finishes.
- Do not use a CPU-only VM for `--dense` (no CUDA).
- Do not skip the `--dev-frac 0.03` / `0.3` gates before paying for full scale.

---

## 10. Vast.ai — cheapest GPU when Jarvis L4/A30 is out of stock

Source of truth: [CLI Hello World](https://docs.vast.ai/cli/hello-world) ·
[docs index](https://docs.vast.ai/llms.txt) · [console](https://cloud.vast.ai).

Marketplace hosts (not a managed datacenter SLA). Prefer **verified** machines. Cheapest path
under ₹500: **1× RTX 3090/4090** at roughly **$0.12–0.22/hr** (~₹10–20/hr) → a 6 h run is
often **₹60–130**.

### Hard filters (do not skip)

| Filter | Why |
|---|---|
| `gpu_name=RTX_3090` or `RTX_4090` | 24 GB VRAM — enough for e5-small + country kNN |
| `cpu_ram>=64` | Our OOM wall; many cheap 3090s ship with ~24–32 GB RAM — **reject those** |
| `cpu_cores>=8` | Blocking + rapidfuzz need CPU |
| `disk_space>=150` | Dataset + work peak ~60 GB + headroom |
| `verified=true` | Host identity-checked by Vast |
| `rentable=true` | Actually available now |
| `direct_port_count>=1` | Direct SSH (lower latency than proxy) |
| `num_gpus=1` | One GPU is enough |

### One-time setup (local Mac)

```bash
# install CLI — https://docs.vast.ai/cli/hello-world
curl -fsSL https://vast.ai/install.sh | bash
# or: pip install vastai

# API key from https://cloud.vast.ai/manage-keys/  (+New — shown once)
vastai set api-key YOUR_API_KEY_HERE
vastai show user

# register SSH key BEFORE first create (applied at container creation)
vastai create ssh-key ~/.ssh/id_ed25519.pub
# or: vastai create ssh-key   # generates one
```

### Search → rent → SSH

```bash
# best value among verified 3090s with enough RAM/disk
vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 cpu_ram>=64 cpu_cores>=8 disk_space>=150 \
   verified=true direct_port_count>=1 rentable=true' \
  -o 'dph+'

# if empty, try RTX_4090 with the same RAM/disk filters
# note the offer ID from the first column

vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  --disk 160 \
  --ssh --direct

# poll until status=running (loading → running; destroy if exited/offline)
vastai show instance INSTANCE_ID

vastai ssh-url INSTANCE_ID
# then: ssh root@HOST -p PORT
```

Notes from their docs:

- Storage bills from **create**; GPU bills from **running**.
- Boot is usually **1–5 min** (image pull). Poll every 10–30 s; do not spin forever on
  `exited` / `unknown` / `offline` — destroy and pick another offer.
- `--onstart-cmd` max **16 KB**.

### On the instance (our pipeline)

```bash
nvidia-smi
curl -LsSf https://astral.sh/uv/install.sh | sh
# upload repo + dataset (from Mac):
#   vastai copy local:./amazon-ml-challenge-2026/ INSTANCE_ID:/workspace/amazon-ml-challenge-2026/
#   or rsync/scp via the ssh-url host:port
cd /workspace/amazon-ml-challenge-2026/code/business_entity_resolution
uv venv .venv -p 3.11
.venv/bin/uv pip install -r requirements.txt
.venv/bin/uv pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/uv pip install -r requirements-dense.txt
.venv/bin/python tests/smoke_test.py
# full run only after promotion gate
```

Torch index: match the image CUDA (example image is **12.4** → `cu124`). If `nvidia-smi`
shows a driver that needs 12.8, switch image/wheels accordingly.

### Copy results home, then stop billing

```bash
vastai copy INSTANCE_ID:/workspace/.../work/models/report.json local:./report.json
vastai copy INSTANCE_ID:/workspace/.../output/ local:./output/

vastai destroy instance INSTANCE_ID   # stops ALL billing
# or: vastai stop instance INSTANCE_ID  # pauses GPU; disk still bills
```

### What not to do

- Do not rent the cheapest 3090 with **&lt;64 GB RAM**.
- Do not leave instances in `loading`/`exited` — disk still costs money.
- Do not use interruptible/spot for a multi-hour full run unless you checkpoint stage caches.
- Prefer Vast when Jarvis **L4/A30** has no stock; prefer Jarvis L4 when it *is* in stock
  (INR invoice, less host lottery).
