"""Optional GPU/MPS pass: multilingual sentence embeddings + top-k dense retrieval.

Model: intfloat/multilingual-e5-small (MIT, 118M params) - reads Devanagari, Kannada,
Tamil, Telugu, Bengali, Gujarati and French natively, so it recovers cross-script
matches that transliteration + keys miss. Enabled with --dense. Embeddings are
L2-normalised fp16 memmaps (N x 384 x 2 bytes) cached on disk.

Encode path: hash unique `name|addr` strings, embed once, scatter back (big win when
duplicates are common). Crash-safe: resumes from `emb_*.npy.tmp` + `.progress`.
Multi-GPU CUDA: one SentenceTransformer per device, unique texts sharded in waves.

kNN: FAISS IndexFlatIP when available (exact cosine on L2-normalised vectors), else
chunked torch gemm. Same top-k semantics either way.
"""
from __future__ import annotations

import hashlib
import os
import time

# Must be set before the first `import torch` in this process: on Windows, PyTorch's
# cu12x wheels ship multi-GB CUDA runtime DLLs (curand/cublas/nvJitLink) whose eager
# load can fail with "[WinError 1455] The paging file is too small" on machines with a
# system-managed pagefile, even with ample RAM. Lazy module loading defers per-kernel
# loading and avoids the up-front commit spike. Harmless on Linux/macOS (ignored) and
# a no-op if the caller already set it.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# Prefer higher MPS memory use before reclaim (helps large encode/knn batches on unified RAM).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np

from utils import LOG, timed


def _pick_device():
    """Prefer CUDA, then Apple MPS, else CPU. Returns (device_str, use_fp16)."""
    import torch
    if torch.cuda.is_available():
        return "cuda", True
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def _cuda_device_count() -> int:
    import torch
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def _texts(df):
    n = df["business_name"].astype(str).tolist()
    a = df["business_address"].astype(str).tolist()
    return [f"query: {x} | {y}" for x, y in zip(n, a)]


def _load_st_model(cfg, device: str, use_fp16: bool):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.dense_model, device=device)
    model.max_seq_length = cfg.dense_max_len
    model.eval()
    if use_fp16 and device.startswith("cuda"):
        model.half()
    return model


def _best_batch(model, texts_probe, candidates, dev: str) -> int:
    """Pick the fastest encode batch that does not OOM (short probe)."""
    import torch
    best_b, best_r = candidates[0], -1.0
    probe = texts_probe[: max(candidates)]
    # warmup
    with torch.inference_mode():
        model.encode(probe[: min(128, len(probe))], batch_size=min(128, candidates[0]),
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    for b in candidates:
        if b > len(probe):
            continue
        try:
            if dev == "mps":
                torch.mps.empty_cache()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.encode(probe, batch_size=b, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            rate = len(probe) / max(1e-6, time.perf_counter() - t0)
            LOG.info("    batch-tune %d -> %.0f rows/s", b, rate)
            if rate > best_r:
                best_r, best_b = rate, b
        except Exception as e:  # noqa: BLE001 — OOM / MPS abort
            LOG.info("    batch-tune %d failed (%s); stopping ramp", b, type(e).__name__)
            break
    return best_b


def _encode_unique_parallel(models, uniq, mm_u, resume_from, nu, batch, step, prog,
                            t_start, *, is_mps: bool = False):
    """Encode uniq[resume_from:nu] into mm_u using one SentenceTransformer per device."""
    import torch
    from concurrent.futures import ThreadPoolExecutor

    ngpu = len(models)

    def _encode_range(model, s0: int, s1: int):
        with torch.inference_mode():
            e = model.encode(
                uniq[s0:s1],
                batch_size=batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return s0, e

    wave = step * ngpu
    for wave_i, s in enumerate(range(resume_from, nu, wave)):
        jobs = []
        for g in range(ngpu):
            s0 = s + g * step
            if s0 >= nu:
                break
            s1 = min(nu, s0 + step)
            jobs.append((g, s0, s1))

        if ngpu == 1:
            results = [_encode_range(models[0], jobs[0][1], jobs[0][2])]
        else:
            with ThreadPoolExecutor(max_workers=ngpu) as pool:
                futs = [pool.submit(_encode_range, models[g], s0, s1) for g, s0, s1 in jobs]
                results = [f.result() for f in futs]

        for s0, e in results:
            mm_u[s0:s0 + len(e)] = e.astype(np.float16)
        done = min(nu, s + wave)
        _save_progress(prog, done)
        elapsed = max(1e-6, time.perf_counter() - t_start)
        session = done - resume_from
        LOG.info("    unique-encoded %d / %d (%.0f%%)  %.0f uniq/s  (ngpu=%d)",
                 done, nu, 100.0 * done / nu, session / elapsed, ngpu)
        if is_mps and (wave_i + 1) % 8 == 0:
            torch.mps.empty_cache()


def build_embeddings(cfg, split: str, s1, idx):
    """Returns (emb_s1, emb_idx) as read-only fp16 memmaps."""
    from prepare import split_dir
    d = split_dir(cfg, split)
    out = []
    for tag, df in (("s1", s1), ("idx", idx)):
        path = os.path.join(d, f"emb_{tag}.npy")
        if not (os.path.exists(path) and not cfg.force):
            _encode_to(path, _texts(df), cfg)
        out.append(np.load(path, mmap_mode="r"))
    return tuple(out)


def _unique_order(texts: list[str]):
    """First-occurrence unique strings + inverse map (row -> unique index)."""
    index: dict[str, int] = {}
    uniq: list[str] = []
    inv = np.empty(len(texts), dtype=np.int32)
    for i, t in enumerate(texts):
        j = index.get(t)
        if j is None:
            j = len(uniq)
            index[t] = j
            uniq.append(t)
        inv[i] = j
    return uniq, inv


def _progress_path(tmp: str) -> str:
    return tmp + ".progress"


def _load_progress(prog: str) -> int:
    if not os.path.exists(prog):
        return 0
    try:
        with open(prog, "r", encoding="utf-8") as f:
            return max(0, int(f.read().strip() or "0"))
    except (ValueError, OSError):
        return 0


def _save_progress(prog: str, done: int) -> None:
    with open(prog, "w", encoding="utf-8") as f:
        f.write(str(int(done)))


def _encode_to(path, texts, cfg):
    """Encode corpus → fp16 memmap with unique-text dedup + crash resume.

    On multi-GPU CUDA boxes (e.g. Kaggle T4×2), unique texts are sharded across
    devices with one SentenceTransformer per GPU. Single-GPU / MPS / CPU unchanged.
    """
    import torch

    ncpu = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(ncpu)
    os.environ.setdefault("OMP_NUM_THREADS", str(ncpu))
    os.environ.setdefault("MKL_NUM_THREADS", str(ncpu))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(ncpu))

    n = len(texts)
    uniq, inv = _unique_order(texts)
    nu = len(uniq)
    LOG.info("    dense dedup: %d rows -> %d unique texts (%.1f%%)",
             n, nu, 100.0 * nu / max(1, n))

    dev, use_fp16 = _pick_device()
    ngpu = _cuda_device_count() if dev == "cuda" else 1
    ngpu = max(1, ngpu)
    if dev == "cuda" and ngpu > 1:
        LOG.info("    multi-GPU encode: %d CUDA devices", ngpu)
        devices = [f"cuda:{g}" for g in range(ngpu)]
    else:
        devices = [dev]

    models = [_load_st_model(cfg, d, use_fp16) for d in devices]
    model0 = models[0]

    base = int(cfg.dense_batch)
    if dev == "cuda":
        # A100 80GB: e5-small is tiny — probe up through 4096.
        cands = sorted({max(base, 1024), 1024, 1536, 2048, 3072, 4096})
    elif dev == "mps":
        # Empirically ~256 is the knee on M5 Air; 1024 hung. Probe around the knee.
        cands = [128, 256, 384, 512, 640]
    else:
        cands = [64, 128, 256]

    if getattr(cfg, "dense_batch_fixed", False):
        batch = int(cfg.dense_batch)
        LOG.info("    pinned encode batch=%d on %s (probe skipped)", batch, devices)
    else:
        probe_n = min(nu, max(cands) * 2, 8192 if dev == "cuda" else 2048)
        batch = _best_batch(model0, uniq[:probe_n], cands, dev) if nu >= 256 else cands[0]
        LOG.info("    selected encode batch=%d on %s", batch, devices)

    dim = model0.get_sentence_embedding_dimension()
    tmp_u = path + ".uniq.tmp.npy"
    prog = _progress_path(tmp_u)
    # Fingerprint so a resumed file matches this unique set.
    finger = hashlib.sha1(("\n".join(uniq[:100]) + f"\n#n={nu}\n#dim={dim}").encode()).hexdigest()[:16]
    meta = path + ".uniq.meta"
    resume_from = 0
    if (os.path.exists(tmp_u) and os.path.exists(meta) and not cfg.force
            and open(meta, encoding="utf-8").read().strip() == finger):
        resume_from = _load_progress(prog)
        LOG.info("    dense resume: continuing unique encode from %d / %d", resume_from, nu)
        mm_u = np.lib.format.open_memmap(tmp_u, mode="r+")
        if mm_u.shape != (nu, dim):
            del mm_u
            resume_from = 0
    if resume_from == 0:
        mm_u = np.lib.format.open_memmap(tmp_u, mode="w+", dtype=np.float16, shape=(nu, dim))
        with open(meta, "w", encoding="utf-8") as f:
            f.write(finger)

    # Larger encode steps on CUDA — less Python overhead, VRAM absorbs it.
    step = max(batch * (32 if dev == "cuda" else 16), 8192)
    t_start = time.perf_counter()
    with timed(f"dense:encode-unique {os.path.basename(path)} uniq={nu:,}/{n:,} "
               f"on {devices} batch={batch}"):
        _encode_unique_parallel(
            models, uniq, mm_u, resume_from, nu, batch, step, prog, t_start,
            is_mps=(dev == "mps"),
        )
    mm_u.flush()

    # Scatter unique → full row memmap.
    tmp = path + ".tmp.npy"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, dim))
    with timed(f"dense:scatter {os.path.basename(path)} n={n:,}"):
        # chunked take to bound RAM
        scat_step = 1_000_000
        for s in range(0, n, scat_step):
            e = min(n, s + scat_step)
            mm[s:e] = mm_u[inv[s:e]]
    mm.flush()
    del mm, mm_u
    for p in (tmp_u, prog, meta):
        try:
            os.remove(p)
        except OSError:
            pass
    os.replace(tmp, path)


def _knn_faiss(emb_q, emb_i, q_rows, i_rows, k):
    """Exact IP top-k via FAISS (CPU or GPU). Same answers as torch on L2-normed vectors."""
    import faiss
    q_rows = np.asarray(q_rows)
    i_rows = np.asarray(i_rows)
    nq, ni = len(q_rows), len(i_rows)
    k = int(min(k, ni))
    I = np.ascontiguousarray(emb_i[i_rows], dtype=np.float32)
    faiss.normalize_L2(I)
    cpu_index = faiss.IndexFlatIP(I.shape[1])
    backend = "cpu"
    index = cpu_index
    try:
        # faiss-gpu / conda faiss with CUDA — no-op failure on faiss-cpu wheels
        ngpu = faiss.get_num_gpus()
        if ngpu > 0:
            res = faiss.StandardGpuResources()
            # Cap temp memory so we don't grab the whole A100 for one country partition.
            res.setTempMemory(2 * 1024 ** 3)
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            backend = f"gpu0/{ngpu}"
    except Exception as e:  # noqa: BLE001
        LOG.info("    faiss GPU unavailable (%s); using CPU IndexFlatIP", type(e).__name__)
        index = cpu_index
    index.add(I)
    Q = np.ascontiguousarray(emb_q[q_rows], dtype=np.float32)
    faiss.normalize_L2(Q)
    # Batched queries — don't put all Q in one shot on huge countries.
    q_batch = 65_536 if backend.startswith("gpu") else min(nq, 262_144)
    ss_parts, ix_parts = [], []
    with timed(f"dense:knn-faiss nq={nq:,} ni={ni:,} k={k} backend={backend} qbatch={q_batch}"):
        for s0 in range(0, nq, q_batch):
            s1 = min(nq, s0 + q_batch)
            D, Ix = index.search(Q[s0:s1], k)
            ss_parts.append(D)
            ix_parts.append(Ix)
    ss = np.concatenate(ss_parts, axis=0)
    ix = np.concatenate(ix_parts, axis=0)
    qq = np.repeat(np.arange(nq, dtype=np.int32), k)
    ii = ix.reshape(-1).astype(np.int32)
    sc = ss.reshape(-1).astype(np.float32)
    m = (ii >= 0) & (sc > -1.5)
    return qq[m], ii[m], sc[m]


def _knn_torch(emb_q, emb_i, q_rows, i_rows, k):
    """Exact top-k cosine via chunked gemm (CUDA / MPS / CPU)."""
    import torch
    dev_s, _ = _pick_device()
    dev = torch.device(dev_s)
    dt = torch.float16 if dev_s == "cuda" else torch.float32
    nq, ni = len(q_rows), len(i_rows)
    k = int(min(k, ni))

    if dev_s == "cuda":
        free, _ = torch.cuda.mem_get_info()
        # A100 80GB: use most of free VRAM for larger gemm tiles.
        mem_frac_i, mem_frac_q = 0.70, 0.40
        qc_cap = 32_768
    elif dev_s == "mps":
        free = 6 * 2**30
        mem_frac_i, mem_frac_q = 0.45, 0.25
        qc_cap = 8192
    else:
        free = 3 * 2**30
        mem_frac_i, mem_frac_q = 0.45, 0.25
        qc_cap = 8192
    dim = emb_i.shape[1]
    bpe = 2 if dt == torch.float16 else 4
    ic = int(max(50_000, min(ni, mem_frac_i * free / (dim * bpe))))
    qc = int(max(128, min(qc_cap, mem_frac_q * free / (max(ic, 1) * bpe))))
    i_rows = np.asarray(i_rows)
    q_rows = np.asarray(q_rows)

    with timed(f"dense:knn nq={nq:,} ni={ni:,} k={k} ichunk={ic:,} qchunk={qc} on {dev_s}"):
        with torch.inference_mode():
            best_v = torch.full((nq, k), -2.0, dtype=torch.float32, device=dev)
            best_i = torch.zeros((nq, k), dtype=torch.int64, device=dev)
            for i0 in range(0, ni, ic):
                I = torch.as_tensor(
                    np.ascontiguousarray(emb_i[i_rows[i0:i0 + ic]]),
                    device=dev, dtype=dt,
                )
                kk = min(k, I.shape[0])
                for q0 in range(0, nq, qc):
                    Q = torch.as_tensor(
                        np.ascontiguousarray(emb_q[q_rows[q0:q0 + qc]]),
                        device=dev, dtype=dt,
                    )
                    v, ix = torch.topk(Q @ I.T, kk, dim=1)
                    v = v.float()
                    ix = ix + i0
                    cv = torch.cat([best_v[q0:q0 + qc], v], 1)
                    ci = torch.cat([best_i[q0:q0 + qc], ix], 1)
                    tv, tix = torch.topk(cv, k, dim=1)
                    best_v[q0:q0 + qc] = tv
                    best_i[q0:q0 + qc] = torch.gather(ci, 1, tix)
                del I
                if dev_s == "cuda":
                    torch.cuda.empty_cache()
            v = best_v.cpu().numpy()
            ix = best_i.cpu().numpy()
            if dev_s == "mps":
                torch.mps.empty_cache()

    qq = np.repeat(np.arange(nq, dtype=np.int32), k)
    ii = ix.reshape(-1).astype(np.int32)
    ss = v.reshape(-1).astype(np.float32)
    m = ss > -1.5
    return qq[m], ii[m], ss[m]


def knn_topk(emb_q, emb_i, q_rows, i_rows, k):
    """Exact top-k cosine for rows q_rows of emb_q against rows i_rows of emb_i.

    Preference: FAISS GPU → FAISS CPU → torch gemm.
    MPS: torch only (FAISS+unified memory OOM'd on Mac).
    """
    nq, ni = len(q_rows), len(i_rows)
    k = int(min(k, ni))
    if nq == 0 or k == 0:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32))
    dev_s, _ = _pick_device()
    if dev_s != "mps":
        try:
            import faiss  # noqa: F401
            return _knn_faiss(emb_q, emb_i, q_rows, i_rows, k)
        except Exception as e:  # noqa: BLE001
            LOG.info("    faiss unavailable (%s); using torch knn", type(e).__name__)
    return _knn_torch(emb_q, emb_i, q_rows, i_rows, k)
