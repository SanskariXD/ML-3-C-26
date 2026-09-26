"""Stage 0: read raw TSVs -> normalise -> country partitions -> parquet cache.

Outputs (work/<split>/):
  s1.parquet, idx.parquet   normalised records (idx = S2 ++ S3, with src column)
  partitions.pkl            {country_key: (s1_rows[int32], idx_rows[int32])}
  gt.npz (train only)       ground-truth pairs as (s1_row, idx_row) int32 arrays
"""
from __future__ import annotations

import os
import pickle
import zlib

import numpy as np
import pandas as pd

from io_utils import read_ground_truth, read_source
from normalize import normalize_frame
from utils import LOG, n_workers, timed


def split_dir(cfg, split: str) -> str:
    d = os.path.join(cfg.work_dir, split)
    os.makedirs(d, exist_ok=True)
    return d


def _country_key(c: str) -> str:
    return c.strip().casefold()


def _dev_subsample(s1: pd.DataFrame, idx: pd.DataFrame, gt: dict, frac: float):
    """Entity-consistent sub-world: keep whole clusters, keep distractors at same rate."""
    thr = int(frac * 1000)
    keep_s1 = np.fromiter((zlib.crc32(b"dev" + x.encode()) % 1000 < thr for x in s1.entity_id),
                          dtype=bool, count=len(s1))
    s1 = s1[keep_s1].reset_index(drop=True)
    kept = set(s1.entity_id)
    owner = {}
    for s, ms in gt.items():
        for m in ms:
            owner[m] = s
    own = idx.entity_id.map(owner)
    has_owner = own.notna().to_numpy()
    keep_owned = own.isin(kept).to_numpy()
    rnd = np.fromiter((zlib.crc32(b"devx" + x.encode()) % 1000 < thr for x in idx.entity_id),
                      dtype=bool, count=len(idx))
    idx = idx[keep_owned | (~has_owner & rnd)].reset_index(drop=True)
    gt = {s: gt.get(s, []) for s in kept}
    LOG.warning("DEV MODE frac=%.3f -> S1=%d IDX=%d (scores are optimistic: fewer name collisions)",
                frac, len(s1), len(idx))
    return s1, idx, gt


def _gt_arrays(s1: pd.DataFrame, idx: pd.DataFrame, gt: dict):
    """Ground truth -> (s1_row, idx_row) int32 arrays (done on plain object ids)."""
    s1_pos = dict(zip(s1["entity_id"].tolist(), range(len(s1))))
    idx_pos = dict(zip(idx["entity_id"].tolist(), range(len(idx))))
    a, b = [], []
    bad = 0
    for s, ms in gt.items():
        r = s1_pos.get(s)
        if r is None:
            bad += len(ms)
            continue
        for m in ms:
            c = idx_pos.get(m)
            if c is None:
                bad += 1
                continue
            a.append(r)
            b.append(c)
    if bad:
        LOG.warning("ground truth: %d pairs reference unknown ids (ignored)", bad)
    return np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.int32)


_RAW_STR = ["entity_id", "business_name", "business_address", "country"]


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    for c in _RAW_STR:
        df[c] = df[c].astype("string[pyarrow]")
    return df


def prepare(cfg, split: str) -> None:
    d = split_dir(cfg, split)
    done = os.path.join(d, "partitions.pkl")
    if os.path.exists(done) and not cfg.force:
        LOG.info("[prepare:%s] cached", split)
        return
    src_dir = os.path.join(cfg.data_dir, split)
    w = n_workers(cfg.workers)
    with timed(f"prepare:{split}:read"):
        s1 = read_source(os.path.join(src_dir, f"{split}_source1.tsv"))
        s2 = read_source(os.path.join(src_dir, f"{split}_source2.tsv"))
        s3 = read_source(os.path.join(src_dir, f"{split}_source3.tsv"))
        s2["src"] = np.int8(2)
        s3["src"] = np.int8(3)
        # an id can only live in one index source; guard against cross-file duplicates
        dup = s3["entity_id"].isin(set(s2["entity_id"]))
        if dup.any():
            LOG.warning("dropping %d S3 ids that also appear in S2", int(dup.sum()))
            s3 = s3[~dup].reset_index(drop=True)
        gt_arr = None
        if split == "train":
            gt = read_ground_truth(os.path.join(src_dir, "train_ground_truth.tsv"))
            if cfg.dev_frac < 1.0:
                idx_tmp = pd.concat([s2, s3], ignore_index=True)
                s1, idx_tmp, gt = _dev_subsample(s1, idx_tmp, gt, cfg.dev_frac)
                s2 = idx_tmp[idx_tmp["src"] == 2].reset_index(drop=True)
                s3 = idx_tmp[idx_tmp["src"] == 3].reset_index(drop=True)
                del idx_tmp
            gt_arr = _gt_arrays(s1, pd.concat([s2[["entity_id"]], s3[["entity_id"]]],
                                              ignore_index=True), gt)
            del gt

    parts_df = []
    for tag, df in (("S1", s1), ("S2", s2), ("S3", s3)):
        with timed(f"prepare:{split}:normalize {tag} ({len(df):,})"):
            parts_df.append(pd.concat([_compact(df), normalize_frame(df, w)], axis=1))
    s1 = parts_df[0]
    idx = pd.concat(parts_df[1:], ignore_index=True)
    del parts_df, s2, s3
    s1["ckey"] = s1["country"].astype(str).map(_country_key).astype("string[pyarrow]")
    idx["ckey"] = idx["country"].astype(str).map(_country_key).astype("string[pyarrow]")

    with timed(f"prepare:{split}:partitions"):
        parts = {}
        idx_ck = idx["ckey"].astype(object)
        idx_groups = pd.Series(np.arange(len(idx)), dtype=np.int64).groupby(
            idx_ck.to_numpy(), sort=True).indices
        blank_idx = idx_groups.get("", np.zeros(0, dtype=np.int64))
        s1_groups = pd.Series(np.arange(len(s1)), dtype=np.int64).groupby(
            s1["ckey"].astype(object).to_numpy(), sort=True).indices
        for ck, rows in s1_groups.items():
            irows = idx_groups.get(ck, np.zeros(0, dtype=np.int64))
            if ck != "" and len(blank_idx):
                irows = np.union1d(irows, blank_idx)
            parts[ck] = (np.sort(np.asarray(rows, dtype=np.int32)),
                         np.sort(np.asarray(irows, dtype=np.int32)))
            LOG.info("  partition %-12r S1=%9d IDX=%9d", ck, len(rows), len(irows))

    with timed(f"prepare:{split}:save"):
        s1.to_parquet(os.path.join(d, "s1.parquet"), index=False)
        idx.to_parquet(os.path.join(d, "idx.parquet"), index=False)
        if gt_arr is not None:
            np.savez(os.path.join(d, "gt.npz"), s1=gt_arr[0], idx=gt_arr[1])
        with open(done, "wb") as f:
            pickle.dump(parts, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_split(cfg, split: str, columns=None):
    d = split_dir(cfg, split)
    # Arrow-backed strings stay compact; Python objects are materialised per partition only
    s1 = pd.read_parquet(os.path.join(d, "s1.parquet"), columns=columns,
                         dtype_backend="pyarrow")
    idx_cols = None if columns is None else list(dict.fromkeys(list(columns) + ["src"]))
    idx = pd.read_parquet(os.path.join(d, "idx.parquet"), columns=idx_cols,
                          dtype_backend="pyarrow")
    with open(os.path.join(d, "partitions.pkl"), "rb") as f:
        parts = pickle.load(f)
    return s1, idx, parts


def load_gt(cfg):
    z = np.load(os.path.join(split_dir(cfg, "train"), "gt.npz"))
    return z["s1"], z["idx"]
