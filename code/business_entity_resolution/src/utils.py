"""Shared helpers: logging, timing, memory probes, deterministic hashing, group ops.

Everything here is dependency-light and deterministic (no Python `hash()`, which is
salted per process on CPython and would make folds/partitions irreproducible).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import time
import zlib

import numpy as np

try:  # psutil is optional; memory numbers degrade to NaN without it
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

LOG = logging.getLogger("ber")


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """UTF-8 safe console + file logging (Windows consoles default to cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    LOG.propagate = False
    return LOG


def rss_gb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / 2**30


def avail_gb() -> float:
    if psutil is None:
        return 8.0
    return psutil.virtual_memory().available / 2**30


def _gpu_stats() -> str:
    """Short CUDA util string, or empty if unavailable.

    Never `import torch` here — that would put torch in sys.modules and trip
    run.py's post-blocking re-exec into an infinite loop on --no-dense runs.
    """
    if "torch" not in sys.modules:
        return ""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated() / 2**30
        return f" | gpu_alloc={alloc:.1f}GB free={free / 2**30:.1f}/{total / 2**30:.1f}GB"
    except Exception:
        return ""


def _cpu_pct() -> str:
    if psutil is None:
        return ""
    try:
        return f" | cpu={psutil.cpu_percent(interval=None):.0f}%"
    except Exception:
        return ""


@contextlib.contextmanager
def timed(name: str):
    t0 = time.perf_counter()
    LOG.info(">> %s", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        LOG.info("<< %s | %.1fs | rss=%.2f GB | avail=%.1f GB%s%s",
                 name, dt, rss_gb(), avail_gb(), _cpu_pct(), _gpu_stats())


def n_workers(requested: int = 0) -> int:
    if requested and requested > 0:
        return int(requested)
    return max(1, (os.cpu_count() or 2))


def crc_fold(ids, n_folds: int = 10, salt: str = "") -> np.ndarray:
    """Deterministic fold id in [0, n_folds) from entity-id strings."""
    s = salt.encode()
    return np.fromiter((zlib.crc32(s + x.encode("utf-8")) % n_folds for x in ids),
                       dtype=np.int8, count=len(ids))


def safe_name(key: str) -> str:
    """Filesystem-safe, collision-free token for an arbitrary country label."""
    base = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")[:40] or "UNK"
    return f"{base}_{zlib.crc32(key.encode('utf-8')) & 0xFFFFFFFF:08x}"


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Vectorised group primitives (groups given as integer ids)                    #
# --------------------------------------------------------------------------- #
def rank_in_sorted_groups(g: np.ndarray) -> np.ndarray:
    """0-based position of each element inside its run, for an array sorted by g."""
    n = len(g)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.empty(n, dtype=bool)
    start[0] = True
    np.not_equal(g[1:], g[:-1], out=start[1:])
    idx = np.arange(n, dtype=np.int64)
    first = np.maximum.accumulate(np.where(start, idx, 0))
    return idx - first


def group_starts(g_sorted: np.ndarray) -> np.ndarray:
    n = len(g_sorted)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])


def group_rank_desc(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """1-based descending rank of v within group g (ties broken by position).

    float32 throughout (not float64): at 60M+ pair-level rows this halves the memory
    of every temporary here, and a rank/similarity value needs nowhere near float64
    precision - a tree model bins it to <=255 levels regardless.
    """
    v = np.nan_to_num(v, nan=-1e9).astype(np.float32, copy=False)
    order = np.lexsort((-v, g))
    r = np.empty(len(g), dtype=np.float32)
    r[order] = rank_in_sorted_groups(g[order]) + 1
    return r


def group_max(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-element max of v over its group (NaN treated as missing). float32 (see
    group_rank_desc)."""
    v = np.nan_to_num(v, nan=-np.inf).astype(np.float32, copy=False)
    order = np.argsort(g, kind="stable")
    gs = g[order]
    st = group_starts(gs)
    mx = np.maximum.reduceat(v[order], st) if len(st) else np.zeros(0, np.float32)
    gid = np.cumsum(np.r_[0, (gs[1:] != gs[:-1]).astype(np.int64)]) if len(gs) else gs
    out = np.empty(len(g), dtype=np.float32)
    out[order] = mx[gid]
    out[~np.isfinite(out)] = np.nan
    return out


def group_top2(g: np.ndarray, v: np.ndarray):
    """Return (rank0, top1_of_group, top2_of_group) aligned to elements. float32 (see
    group_rank_desc).

    rank0 is the 0-based descending rank; top2 is NaN for single-element groups.
    """
    vv = np.nan_to_num(v, nan=-1e9).astype(np.float32, copy=False)
    order = np.lexsort((-vv, g))
    gs = g[order]
    rk = rank_in_sorted_groups(gs)
    st = group_starts(gs)
    sizes = np.diff(np.r_[st, len(gs)])
    top1 = vv[order][st]
    top2 = np.where(sizes > 1, vv[order][np.minimum(st + 1, len(gs) - 1)], np.nan)
    gid = np.repeat(np.arange(len(st)), sizes)
    rank0 = np.empty(len(g), dtype=np.int64)
    rank0[order] = rk
    t1 = np.empty(len(g), dtype=np.float32)
    t2 = np.empty(len(g), dtype=np.float32)
    t1[order] = top1[gid]
    t2[order] = top2[gid]
    t1[t1 <= -1e8] = np.nan
    t2[t2 <= -1e8] = np.nan
    return rank0, t1, t2


def popcount16(bits: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(bits.astype(np.uint16))
    return np.unpackbits(b.view(np.uint8)).reshape(-1, 16).sum(1).astype(np.float32)
