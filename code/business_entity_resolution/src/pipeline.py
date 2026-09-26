"""Training and inference orchestration.

Fold protocol (fold = crc32(S1 entity_id) % 10, i.e. grouped by entity - no leakage
between the candidates of one S1):
  folds 0-2  stage-1 LightGBM, 3-fold CV   -> OOF p1 here, mean-of-3 p1 elsewhere
  folds 3-7  stage-2 LightGBM, 5-fold CV   -> trained on mean-of-3 p1, exactly the
             distribution the test set sees; OOF p2 here -> calibration + decision tuning
  folds 8-9  untouched holdout: final, unbiased F0.5 estimate (mean-of-5 p2, like test)
Blocking and features are computed for ALL train S1 so owner-competition features see
the same density as the test set.
"""
from __future__ import annotations

import gc
import os
import pickle
import time

import numpy as np

from config import FEATURE_CONTRACT, Config
from decide import apply_decision, exclusive_mask, macro_f05, tune
from features import ALL_FEATURES, MONOTONE_UP, feat_path, load_features
from prepare import load_gt, load_split, split_dir
from stage2 import S2_FEATURES, S2_MONOTONE_UP, stage2_matrix
from utils import LOG, crc_fold, load_json, safe_name, save_json, timed

PRED_CHUNK = 2_000_000


def _maybe_delete(cfg: Config, path: str) -> None:
    """Best-effort removal of a large, fully-consumed intermediate (feat_*/s2_*.npy).

    No-op when cfg.keep_intermediates. A memmap's backing file can briefly stay open
    on Windows after `del` until the refcounted mmap object is actually finalized, so
    this forces a collection and retries a few times rather than failing the run over
    a file that a moment later would have deleted cleanly.
    """
    if cfg.keep_intermediates or not os.path.exists(path):
        return
    gc.collect()
    for attempt in range(5):
        try:
            os.remove(path)
            LOG.info("  freed disk: removed %s", path)
            return
        except OSError as exc:
            if attempt == 4:
                LOG.warning("  could not remove %s (%s) - leaving it on disk", path, exc)
                return
            time.sleep(0.2)


def model_dir(cfg):
    d = os.path.join(cfg.work_dir, "models")
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# LightGBM helpers                                                            #
# --------------------------------------------------------------------------- #
def lgb_params(cfg, names, monotone_set):
    p = dict(objective="binary", learning_rate=cfg.lgb_lr, num_leaves=cfg.lgb_leaves,
             min_data_in_leaf=cfg.lgb_min_leaf, feature_fraction=0.8, bagging_fraction=0.75,
             bagging_freq=1, lambda_l1=0.3, lambda_l2=1.5, min_gain_to_split=0.01,
             max_bin=255, num_threads=os.cpu_count() or 4,
             # Histogram GBDT, row-wise layout (wide-enough rows, few columns), all cores.
             seed=cfg.seed, deterministic=True, force_row_wise=True, verbose=-1,
             metric=["binary_logloss", "auc"])
    if cfg.monotone:
        p["monotone_constraints"] = [1 if n in monotone_set else 0 for n in names]
        p["monotone_constraints_method"] = "intermediate"
    return p


def train_cv(X, y, fold, fold_ids, cfg, names, monotone_set, tag):
    """Grouped CV on a single binned Dataset (subset() shares bins -> ~4x less RAM)."""
    import lightgbm as lgb
    params = lgb_params(cfg, names, monotone_set)
    ds = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=True,
                     params={"max_bin": 255, "verbose": -1})
    ds.construct()
    models = {}
    for f in fold_ids:
        tr = np.flatnonzero(fold != f)
        va = np.flatnonzero(fold == f)
        if len(va) == 0 or len(tr) == 0:
            continue
        with timed(f"lgb:{tag}:fold{f} train={len(tr):,} valid={len(va):,}"):
            b = lgb.train(params, ds.subset(tr), num_boost_round=cfg.lgb_rounds,
                          valid_sets=[ds.subset(va)], valid_names=["valid"],
                          callbacks=[lgb.early_stopping(cfg.early_stop, verbose=False),
                                     lgb.log_evaluation(200)])
        path = os.path.join(model_dir(cfg), f"{tag}_f{f}.txt")
        b.save_model(path, num_iteration=b.best_iteration)
        LOG.info("  %s fold %d: best_iter=%d  %s", tag, f, b.best_iteration,
                 {k: round(v, 5) for k, v in b.best_score["valid"].items()})
        models[f] = lgb.Booster(model_file=path)
        imp = sorted(zip(names, b.feature_importance("gain")), key=lambda t: -t[1])[:15]
        LOG.info("  top gain: %s", ", ".join(f"{n}={g:.0f}" for n, g in imp))
    return models


def load_models(cfg, tag):
    import lightgbm as lgb
    out = {}
    for fn in sorted(os.listdir(model_dir(cfg))):
        if fn.startswith(tag + "_f") and fn.endswith(".txt"):
            out[int(fn[len(tag) + 2:-4])] = lgb.Booster(model_file=os.path.join(model_dir(cfg), fn))
    if not out:
        raise FileNotFoundError(f"no {tag} models in {model_dir(cfg)} - run `train` first")
    return out


def predict_rows(models, X, fold=None):
    """Own-fold model for OOF rows (fold in models), mean of all models elsewhere."""
    preds = {f: m.predict(X, num_threads=os.cpu_count() or 4) for f, m in models.items()}
    p = np.mean(np.stack(list(preds.values())), axis=0)
    if fold is not None:
        for f, pf in preds.items():
            m = fold == f
            p[m] = pf[m]
    return p.astype(np.float32)


def _subsample_hard_neg(y, bscore, brank, dcos, max_rows, seed):
    """Keep ALL positives + hardest negatives up to max_rows.

    Hardness ≈ blocking score + dense cos − small*brank. Falls back to random
    among ties. Preferential for BM25/dense/key near-misses over random junk.
    """
    n = len(y)
    if n <= max_rows:
        return np.ones(n, bool)
    y = np.asarray(y)
    pos = y > 0
    n_pos = int(pos.sum())
    if n_pos >= max_rows:
        # Extreme imbalance: keep a random pos subset (should never happen).
        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(pos)
        pick = rng.choice(idx, size=max_rows, replace=False)
        keep = np.zeros(n, bool)
        keep[pick] = True
        LOG.warning("positives %d >= cap %d: random pos subsample", n_pos, max_rows)
        return keep
    n_neg = max_rows - n_pos
    neg = np.flatnonzero(~pos)
    bs = np.nan_to_num(np.asarray(bscore, dtype=np.float32), nan=0.0)
    dc = np.nan_to_num(np.asarray(dcos, dtype=np.float32), nan=0.0)
    br = np.asarray(brank, dtype=np.float32)
    hard = bs[neg] + 0.5 * dc[neg] - 0.01 * br[neg]
    # slight noise so order isn't fully deterministic across identical scores
    rng = np.random.default_rng(seed)
    hard = hard + 1e-6 * rng.random(len(neg), dtype=np.float32)
    if len(neg) <= n_neg:
        keep = np.ones(n, bool)
    else:
        # argpartition: largest n_neg hardness
        part = np.argpartition(-hard, n_neg - 1)[:n_neg]
        keep = pos.copy()
        keep[neg[part]] = True
    LOG.warning("training rows %d > cap %d: keep all %d positives + %d hard negatives "
                "(%.1f%% of rows)", n, max_rows, n_pos, int(keep.sum()) - n_pos,
                100.0 * keep.mean())
    return keep


def _subsample_groups(q_glob, max_rows, seed):
    """Legacy group subsample — kept for callers that lack pair scores."""
    if len(q_glob) <= max_rows:
        return np.ones(len(q_glob), bool)
    ratio = max_rows / len(q_glob)
    u = np.unique(q_glob)
    rng = np.random.default_rng(seed)
    keep_q = u[rng.random(len(u)) < ratio]
    LOG.warning("training rows %d > cap %d: sub-sampling %.1f%% of S1 groups",
                len(q_glob), max_rows, 100 * ratio)
    return np.isin(q_glob, keep_q)


# --------------------------------------------------------------------------- #
# training                                                                    #
# --------------------------------------------------------------------------- #
def _labels(cfg, ck, s1_rows, idx_rows, P, gt_keys, n_idx_total):
    path = os.path.join(split_dir(cfg, "train"), f"y_{safe_name(ck)}.npy")
    if os.path.exists(path) and not cfg.force:
        return np.load(path)
    key = s1_rows[P["q"]].astype(np.int64) * n_idx_total + idx_rows[P["i"]].astype(np.int64)
    y = np.isin(key, gt_keys).astype(np.int8)
    np.save(path, y)
    return y


def train_all(cfg: Config) -> dict:
    from blocking import load_pairs
    s1, idx, parts = load_split(cfg, "train", columns=["entity_id", "country", "ckey",
                                                       "n_core", "a_full", "a_nums"])
    n_s1, n_idx = len(s1), len(idx)
    fold_s1 = crc_fold(s1["entity_id"].tolist())
    gt_s1, gt_idx = load_gt(cfg)
    gt_keys = np.unique(gt_s1.astype(np.int64) * n_idx + gt_idx.astype(np.int64))
    G = np.bincount(gt_s1, minlength=n_s1).astype(np.float64)
    allowed = set(c.casefold() for c in cfg.train_countries) if cfg.train_countries else None
    from decide import adjust_p_housenum, first_housenum_arr
    q_hnum = first_housenum_arr(s1["a_nums"].to_numpy(dtype=object))
    i_hnum = first_housenum_arr(idx["a_nums"].to_numpy(dtype=object))

    # ---------- per-partition metadata ----------
    meta = {}
    for ck, (s1_rows, idx_rows) in parts.items():
        P = load_pairs(cfg, "train", ck)
        y = _labels(cfg, ck, s1_rows, idx_rows, P, gt_keys, n_idx)
        qg = s1_rows[P["q"]]
        meta[ck] = dict(P=P, y=y, qg=qg, ig=idx_rows[P["i"]], fold=fold_s1[qg],
                        use=(allowed is None or ck in allowed))
        LOG.info("  %r pairs=%d positives=%d", ck, len(y), int(y.sum()))

    # ---------- blocking diagnostics ----------
    tot_pos = sum(int(m["y"].sum()) for m in meta.values())
    recall = tot_pos / max(1, len(gt_keys))
    LOG.info("BLOCKING recall ceiling (all train) = %.4f  pairs/S1 = %.1f", recall,
             sum(len(m["y"]) for m in meta.values()) / max(1, n_s1))

    # ---------- stage 1 ----------
    with timed("stage1: assemble matrix"):
        Xs, ys, fs, qs, bss, brs, dcs = [], [], [], [], [], [], []
        for ck, m in meta.items():
            if not m["use"]:
                continue
            sel = np.flatnonzero(np.isin(m["fold"], cfg.stage1_folds))
            if len(sel) == 0:
                continue
            mm = load_features(cfg, "train", ck)
            Xs.append(np.asarray(mm[sel], dtype=np.float32))
            ys.append(m["y"][sel])
            fs.append(m["fold"][sel])
            qs.append(m["qg"][sel])
            P = m["P"]
            bss.append(P["bscore"][sel])
            brs.append(P["brank"][sel])
            dcs.append(P["dcos"][sel] if "dcos" in P else np.full(len(sel), np.nan, np.float32))
        X, y, fold, qg = (np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs),
                          np.concatenate(qs))
        bscore = np.concatenate(bss)
        brank = np.concatenate(brs)
        dcos = np.concatenate(dcs)
        del Xs, bss, brs, dcs
        keep = _subsample_hard_neg(y, bscore, brank, dcos, cfg.max_train_rows, cfg.seed)
        if not keep.all():
            X, y, fold = X[keep], y[keep], fold[keep]
        LOG.info("stage1 matrix %s  pos_rate=%.4f  (%.2f GB)", X.shape, y.mean(), X.nbytes / 2**30)
    m1 = train_cv(X, y, fold, cfg.stage1_folds, cfg, ALL_FEATURES, MONOTONE_UP, "stage1")
    del X

    # ---------- stage-1 inference on all train pairs + stage-2 features ----------
    idx_ncore_all = idx["n_core"].to_numpy(dtype=object)
    idx_afull_all = idx["a_full"].to_numpy(dtype=object)
    for ck, m in meta.items():
        with timed(f"stage1 predict + stage2 features {ck!r}"):
            mm = load_features(cfg, "train", ck)
            n = mm.shape[0]
            p1 = np.empty(n, np.float32)
            for s in range(0, n, PRED_CHUNK):
                blk = np.asarray(mm[s:s + PRED_CHUNK], dtype=np.float32)
                p1[s:s + PRED_CHUNK] = predict_rows(m1, blk, m["fold"][s:s + PRED_CHUNK])
            m["p1"] = p1
            idx_rows = parts[ck][1]
            i_ncore = idx_ncore_all[idx_rows]
            i_afull = idx_afull_all[idx_rows]
            s2_path = os.path.join(split_dir(cfg, "train"), f"s2_{safe_name(ck)}.npy")
            X2 = np.lib.format.open_memmap(s2_path, mode="w+", dtype=np.float16,
                                           shape=(n, len(S2_FEATURES)))
            stage2_matrix(m["P"]["q"].astype(np.int64), m["P"]["i"].astype(np.int64),
                          p1, mm, i_ncore, i_afull, out=X2)
            X2.flush()
            del X2, mm
            # feat_*.npy (fp16, several GB/partition) is not read again in this run;
            # pairs_*.npz (the expensive-to-recompute blocking cache) is always kept.
            _maybe_delete(cfg, feat_path(cfg, "train", ck))

    # ---------- stage 2 ----------
    with timed("stage2: assemble matrix"):
        Xs, ys, fs, qs, bss, brs, dcs = [], [], [], [], [], [], []
        for ck, m in meta.items():
            if not m["use"]:
                continue
            sel = np.flatnonzero(np.isin(m["fold"], cfg.stage2_folds))
            X2 = np.load(os.path.join(split_dir(cfg, "train"), f"s2_{safe_name(ck)}.npy"),
                         mmap_mode="r")
            Xs.append(np.asarray(X2[sel], dtype=np.float32))
            del X2
            ys.append(m["y"][sel])
            fs.append(m["fold"][sel])
            qs.append(m["qg"][sel])
            P = m["P"]
            bss.append(P["bscore"][sel])
            brs.append(P["brank"][sel])
            dcs.append(P["dcos"][sel] if "dcos" in P else np.full(len(sel), np.nan, np.float32))
        X, y, fold, qg = (np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs),
                          np.concatenate(qs))
        bscore = np.concatenate(bss)
        brank = np.concatenate(brs)
        dcos = np.concatenate(dcs)
        del Xs, bss, brs, dcs
        keep = _subsample_hard_neg(y, bscore, brank, dcos, cfg.max_train_rows, cfg.seed + 1)
        if not keep.all():
            X, y, fold = X[keep], y[keep], fold[keep]
    m2 = train_cv(X, y, fold, cfg.stage2_folds, cfg, S2_FEATURES, S2_MONOTONE_UP, "stage2")
    del X

    # ---------- p2 for all train pairs ----------
    for ck, m in meta.items():
        s2_path = os.path.join(split_dir(cfg, "train"), f"s2_{safe_name(ck)}.npy")
        X2 = np.load(s2_path, mmap_mode="r")
        n = X2.shape[0]
        p2 = np.empty(n, np.float32)
        for s in range(0, n, PRED_CHUNK):
            blk = np.asarray(X2[s:s + PRED_CHUNK], dtype=np.float32)
            p2[s:s + PRED_CHUNK] = predict_rows(m2, blk, m["fold"][s:s + PRED_CHUNK])
        m["p2"] = p2
        del X2
        # last use of this partition's stage-2 matrix in this run.
        _maybe_delete(cfg, s2_path)

    # ---------- global arrays ----------
    q = np.concatenate([m["qg"] for m in meta.values()]).astype(np.int64)
    ig = np.concatenate([m["ig"] for m in meta.values()]).astype(np.int64)
    y = np.concatenate([m["y"] for m in meta.values()])
    p1 = np.concatenate([m["p1"] for m in meta.values()])
    p2 = np.concatenate([m["p2"] for m in meta.values()])
    fq = fold_s1[q]
    s2mask_rows = np.isin(fq, cfg.stage2_folds)

    # ---------- calibration (isotonic on OOF p2) ----------
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p2[s2mask_rows], y[s2mask_rows])
    with open(os.path.join(model_dir(cfg), "calibrator.pkl"), "wb") as f:
        pickle.dump(iso, f)
    pc = iso.predict(p2).astype(np.float32)
    pc = adjust_p_housenum(pc, q, ig, q_hnum, i_hnum,
                           boost=float(getattr(cfg, "housenum_boost", 0.0) or 0.0),
                           pen=float(getattr(cfg, "housenum_pen", 0.0) or 0.0))

    # ---------- decision tuning on OOF (folds 3-7), report on holdout (8-9) ----------
    q_tune = np.isin(fold_s1, cfg.stage2_folds)
    q_hold = np.isin(fold_s1, cfg.holdout_folds)
    with timed("decision tuning"):
        best, _ = tune(q, ig, pc, y, G, q_tune, n_s1, cfg.decision)
        best1, _ = tune(q, ig, p1, y, G, q_tune, n_s1, "threshold")
    # record housenum knobs so predict_test reapplies the same adjustment
    best = dict(best)
    best["housenum_boost"] = float(getattr(cfg, "housenum_boost", 0.0) or 0.0)
    best["housenum_pen"] = float(getattr(cfg, "housenum_pen", 0.0) or 0.0)
    sel = apply_decision(q, ig, pc, n_s1, best)
    sel1 = apply_decision(q, ig, p1, n_s1, best1)
    ex = exclusive_mask(ig, pc)
    report = {
        "blocking_recall_all": recall,
        "pairs_per_s1": len(q) / max(1, n_s1),
        "holdout_blocking_recall": float(y[np.isin(fq, cfg.holdout_folds)].sum()
                                         / max(1, G[q_hold].sum())),
        "holdout_oracle_f05": macro_f05(q[y == 1], y[y == 1], G, q_hold),
        "holdout_stage1_f05": macro_f05(q[sel1], y[sel1], G, q_hold),
        "holdout_stage2_f05": macro_f05(q[sel], y[sel], G, q_hold),
        "holdout_stage2_threshold05_f05": macro_f05(q[ex & (pc >= 0.5)], y[ex & (pc >= 0.5)],
                                                    G, q_hold),
        "tuned_f05_oof": macro_f05(q[sel], y[sel], G, q_tune),
        "decision": best,
        "decision_stage1": best1,
        "per_country_holdout_f05": {},
    }
    ck_of_s1 = s1["ckey"].to_numpy(dtype=object)
    for ck in parts:
        qm = q_hold & (ck_of_s1 == ck)
        report["per_country_holdout_f05"][ck] = macro_f05(q[sel], y[sel], G, qm)
    for k, v in report.items():
        LOG.info("REPORT %-32s %s", k, v)
    save_json(report, os.path.join(model_dir(cfg), "report.json"))
    contract = {k: getattr(cfg, k) for k in FEATURE_CONTRACT}
    save_json({"contract": contract, "decision": best, "stage1_features": ALL_FEATURES,
               "stage2_features": S2_FEATURES, "config": cfg.to_dict()},
              os.path.join(model_dir(cfg), "meta.json"))
    return report


# --------------------------------------------------------------------------- #
# inference                                                                   #
# --------------------------------------------------------------------------- #
def enforce_contract(cfg: Config) -> dict:
    meta = load_json(os.path.join(model_dir(cfg), "meta.json"))
    for k, v in meta["contract"].items():
        if getattr(cfg, k) != v:
            LOG.warning("config %s=%r differs from training (%r) - using training value",
                        k, getattr(cfg, k), v)
            setattr(cfg, k, v)
    if meta["stage1_features"] != ALL_FEATURES or meta["stage2_features"] != S2_FEATURES:
        raise RuntimeError("feature list changed since training - retrain")
    return meta


def predict_test(cfg: Config):
    from blocking import load_pairs
    from io_utils import (CAND_HEADER, MATCH_HEADER, run_official_validator, self_check,
                          write_id_lists)
    meta = load_json(os.path.join(model_dir(cfg), "meta.json"))
    m1, m2 = load_models(cfg, "stage1"), load_models(cfg, "stage2")
    with open(os.path.join(model_dir(cfg), "calibrator.pkl"), "rb") as f:
        iso = pickle.load(f)
    s1, idx, parts = load_split(cfg, "test", columns=["entity_id", "n_core", "a_full", "a_nums"])
    n_s1 = len(s1)
    from decide import adjust_p_housenum, first_housenum_arr
    q_hnum = first_housenum_arr(s1["a_nums"].to_numpy(dtype=object))
    i_hnum = first_housenum_arr(idx["a_nums"].to_numpy(dtype=object))
    idx_ncore_all = idx["n_core"].to_numpy(dtype=object)
    idx_afull_all = idx["a_full"].to_numpy(dtype=object)
    Q, I, PC = [], [], []
    for ck, (s1_rows, idx_rows) in parts.items():
        P = load_pairs(cfg, "test", ck)
        n = len(P["q"])
        if n == 0:
            continue
        with timed(f"predict {ck!r} pairs={n:,}"):
            mm = load_features(cfg, "test", ck)
            p1 = np.empty(n, np.float32)
            for s in range(0, n, PRED_CHUNK):
                p1[s:s + PRED_CHUNK] = predict_rows(m1, np.asarray(mm[s:s + PRED_CHUNK],
                                                                   dtype=np.float32))
            s2_path = os.path.join(split_dir(cfg, "test"), f"s2_{safe_name(ck)}.npy")
            X2 = np.lib.format.open_memmap(s2_path, mode="w+", dtype=np.float16,
                                           shape=(n, len(S2_FEATURES)))
            stage2_matrix(P["q"].astype(np.int64), P["i"].astype(np.int64), p1, mm,
                          idx_ncore_all[idx_rows], idx_afull_all[idx_rows], out=X2)
            p2 = np.empty(n, np.float32)
            for s in range(0, n, PRED_CHUNK):
                blk = np.asarray(X2[s:s + PRED_CHUNK], dtype=np.float32)
                p2[s:s + PRED_CHUNK] = predict_rows(m2, blk)
            del X2, mm
            # single-pass inference: neither file is read again once p2 is computed.
            _maybe_delete(cfg, feat_path(cfg, "test", ck))
            _maybe_delete(cfg, s2_path)
        Q.append(s1_rows[P["q"]].astype(np.int64))
        I.append(idx_rows[P["i"]].astype(np.int64))
        PC.append(iso.predict(p2).astype(np.float32))
    q = np.concatenate(Q) if Q else np.zeros(0, np.int64)
    ig = np.concatenate(I) if I else np.zeros(0, np.int64)
    pc = np.concatenate(PC) if PC else np.zeros(0, np.float32)
    dcfg = meta["decision"]
    pc = adjust_p_housenum(
        pc, q, ig, q_hnum, i_hnum,
        boost=float(dcfg.get("housenum_boost", getattr(cfg, "housenum_boost", 0.0)) or 0.0),
        pen=float(dcfg.get("housenum_pen", getattr(cfg, "housenum_pen", 0.0)) or 0.0),
    )
    sel = apply_decision(q, ig, pc, n_s1, dcfg)
    LOG.info("test: %d candidate pairs, %d selected, %.1f%% S1 with >=1 match",
             len(q), int(sel.sum()), 100 * len(np.unique(q[sel])) / max(1, n_s1))

    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    idx_ids = idx["entity_id"].to_numpy(dtype=object)
    os.makedirs(cfg.out_dir, exist_ok=True)
    mpath = os.path.join(cfg.out_dir, "matching_results.tsv")
    cpath = os.path.join(cfg.out_dir, "candidate_pairs.tsv")
    write_id_lists(mpath, MATCH_HEADER, s1_ids, q[sel], idx_ids[ig[sel]], pc[sel])
    write_id_lists(cpath, CAND_HEADER, s1_ids, q, idx_ids[ig], pc)
    required = s1_ids.tolist()
    for path, header in ((mpath, MATCH_HEADER), (cpath, CAND_HEADER)):
        errs = self_check(path, header, required)
        if errs:
            raise RuntimeError(f"{path} failed self-check: {errs[:5]}")
        LOG.info("self-check PASS: %s", path)
    np.savez(os.path.join(split_dir(cfg, "test"), "final_scores.npz"), q=q, i=ig, p=pc, sel=sel)
    rc = run_official_validator(cfg.validator, mpath, cpath,
                                os.path.join(cfg.data_dir, "test"))
    if rc != 0:
        raise RuntimeError("official validator FAILED - see log")
    return mpath, cpath
