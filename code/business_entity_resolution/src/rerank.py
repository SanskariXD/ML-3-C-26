"""Stage 3 (optional): a fine-tuned cross-encoder that re-scores the SHORTLIST only.

Why: LightGBM on handcrafted string-similarity features tops out at what those
features can express. A cross-encoder transformer reads both records' raw text
directly (Ditto, Li et al. 2020, VLDB - "Deep Entity Matching with Pre-trained
Language Models") and can pick up semantic/typo/transliteration equivalence the
handcrafted features miss - the paper reports up to +29 F1 over prior feature-based
matchers on public benchmarks. Its cost is per-pair inference time, so it is not run
over the full candidate universe (100M+ pairs here): it re-scores only the SHORTLIST
- the pairs stage-2 + decision already selected for submission (order of a few
pairs per S1, a few million total) - where it can only ever help precision, which is
what F0.5 weights twice as heavily as recall. This mirrors how Ditto and cross-encoder
re-rankers are used in production retrieve-then-rerank pipelines generally.

Base model: intfloat/multilingual-e5-small (MIT, 118M params, already used for the
dense blocking pass) repurposed as a binary cross-encoder - well under the 8B/
permissive-licence constraint. Training data: ground-truth positives plus hard
negatives mined from the blocking candidates (same-S1 non-matches, prioritised by
blocking score) - no external data, consistent with the fair-play rules.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils import LOG, crc_fold, timed

MAX_FIELD = 100  # characters kept per name/address in the serialised pair


def _clip(s: str) -> str:
    s = "" if s is None else str(s)
    return s[:MAX_FIELD]


def build_texts(names: np.ndarray, addrs: np.ndarray, countries: np.ndarray):
    """One serialised string per record: 'name: X address: Y country: Z'."""
    out = np.empty(len(names), dtype=object)
    for i in range(len(names)):
        out[i] = f"name: {_clip(names[i])} address: {_clip(addrs[i])} country: {_clip(countries[i])}"
    return out


def _mine_pairs(cfg, split: str, n_pos: int, neg_per_pos: int, seed: int):
    """Sample (s1_row, idx_row, label) training triples for the reranker.

    Positives: ground-truth matches. Hard negatives: OTHER candidates blocking found
    for the same S1 that are not the true match, weighted toward higher blocking score
    (bscore) so the reranker specifically learns to separate near-misses, not just
    obviously unrelated records - the case that actually matters at decision time.
    """
    from blocking import load_pairs
    from prepare import load_gt, load_split

    s1, idx, parts = load_split(cfg, split, columns=["entity_id", "business_name",
                                                     "business_address", "country"])
    gt_s1, gt_idx = load_gt(cfg)
    n_idx = len(idx)
    pos_keys = set((gt_s1.astype(np.int64) * n_idx + gt_idx.astype(np.int64)).tolist())

    rng = np.random.default_rng(seed)
    pos_rows_q, pos_rows_i = [], []
    neg_rows_q, neg_rows_i = [], []
    for ck, (s1_rows, idx_rows) in parts.items():
        P = load_pairs(cfg, split, ck)
        if len(P["q"]) == 0:
            continue
        gq = s1_rows[P["q"]].astype(np.int64)
        gi = idx_rows[P["i"]].astype(np.int64)
        key = gq * n_idx + gi
        is_pos = np.fromiter((k in pos_keys for k in key.tolist()), dtype=bool, count=len(key))
        pos_rows_q.append(gq[is_pos])
        pos_rows_i.append(gi[is_pos])
        # hard negatives: same partition, not a match, weighted by bscore
        neg_idx = np.flatnonzero(~is_pos)
        if len(neg_idx) == 0:
            continue
        w = P["bscore"][neg_idx].astype(np.float64)
        w = w - w.min() + 1e-3
        take = min(len(neg_idx), int(len(pos_rows_q[-1]) * neg_per_pos * 1.5) + 100)
        chosen = rng.choice(neg_idx, size=min(take, len(neg_idx)), replace=False,
                            p=w / w.sum())
        neg_rows_q.append(gq[chosen])
        neg_rows_i.append(gi[chosen])

    pos_q = np.concatenate(pos_rows_q) if pos_rows_q else np.zeros(0, np.int64)
    pos_i = np.concatenate(pos_rows_i) if pos_rows_i else np.zeros(0, np.int64)
    neg_q = np.concatenate(neg_rows_q) if neg_rows_q else np.zeros(0, np.int64)
    neg_i = np.concatenate(neg_rows_i) if neg_rows_i else np.zeros(0, np.int64)

    if len(pos_q) > n_pos:
        sel = rng.choice(len(pos_q), n_pos, replace=False)
        pos_q, pos_i = pos_q[sel], pos_i[sel]
    n_neg = min(len(neg_q), len(pos_q) * neg_per_pos)
    if len(neg_q) > n_neg:
        sel = rng.choice(len(neg_q), n_neg, replace=False)
        neg_q, neg_i = neg_q[sel], neg_i[sel]

    q = np.concatenate([pos_q, neg_q])
    i = np.concatenate([pos_i, neg_i])
    y = np.concatenate([np.ones(len(pos_q), np.int8), np.zeros(len(neg_q), np.int8)])
    LOG.info("reranker training pairs: %d positive, %d negative", len(pos_q), len(neg_q))

    names_q = s1["business_name"].to_numpy(dtype=object)[q]
    addrs_q = s1["business_address"].to_numpy(dtype=object)[q]
    ctry_q = s1["country"].to_numpy(dtype=object)[q]
    names_i = idx["business_name"].to_numpy(dtype=object)[i]
    addrs_i = idx["business_address"].to_numpy(dtype=object)[i]
    ctry_i = idx["country"].to_numpy(dtype=object)[i]
    txt_a = build_texts(names_q, addrs_q, ctry_q)
    txt_b = build_texts(names_i, addrs_i, ctry_i)

    fold = crc_fold(s1["entity_id"].to_numpy(dtype=object)[q].tolist())
    return txt_a, txt_b, y, fold


def train_reranker(cfg, base_model: str = "intfloat/multilingual-e5-small",
                   n_pos: int = 300_000, neg_per_pos: int = 2, epochs: int = 1,
                   batch_size: int = 64, train_folds=None, out_dir: str | None = None):
    """Fine-tune a small cross-encoder on mined (positive, hard-negative) pairs.

    Only rows whose fold is in `train_folds` are used, so the SAME holdout folds
    (8, 9) used to report the GBM pipeline's numbers stay untouched here too -
    `evaluate_reranker` below gives an honest, comparable holdout F0.5 delta.
    """
    import torch
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    with timed("rerank:mine training pairs"):
        txt_a, txt_b, y, fold = _mine_pairs(cfg, "train", n_pos, neg_per_pos, cfg.seed)
    train_folds = train_folds if train_folds is not None else list(range(8))
    m = np.isin(fold, train_folds)
    txt_a, txt_b, y = txt_a[m], txt_b[m], y[m]
    LOG.info("reranker: training on %d pairs (folds %s)", len(y), train_folds)

    from sentence_transformers import InputExample
    examples = [InputExample(texts=[a, b], label=float(l))
               for a, b, l in zip(txt_a.tolist(), txt_b.tolist(), y.tolist())]
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(examples)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(base_model, num_labels=1, max_length=96, device=dev)
    from torch.utils.data import DataLoader
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    with timed(f"rerank:fine-tune n={len(examples):,} epochs={epochs} on {dev}"):
        model.fit(train_dataloader=loader, epochs=epochs, warmup_steps=min(1000, len(loader) // 10),
                  show_progress_bar=False, output_path=None)

    out_dir = out_dir or os.path.join(cfg.work_dir, "models", "reranker")
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_dir)
    LOG.info("reranker saved to %s", out_dir)
    return out_dir


def load_reranker(model_dir: str):
    import torch
    from sentence_transformers import CrossEncoder
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(model_dir, max_length=96, device=dev)


def score_shortlist(model, names_q, addrs_q, ctry_q, names_i, addrs_i, ctry_i,
                    batch_size: int = 256) -> np.ndarray:
    """Cross-encoder probability for a (typically small) shortlist of pairs."""
    if len(names_q) == 0:
        return np.zeros(0, dtype=np.float32)
    txt_a = build_texts(np.asarray(names_q, dtype=object), np.asarray(addrs_q, dtype=object),
                        np.asarray(ctry_q, dtype=object))
    txt_b = build_texts(np.asarray(names_i, dtype=object), np.asarray(addrs_i, dtype=object),
                        np.asarray(ctry_i, dtype=object))
    pairs = list(zip(txt_a.tolist(), txt_b.tolist()))
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False,
                           apply_softmax=False, convert_to_numpy=True)
    # CrossEncoder(num_labels=1) trained with BCE outputs a raw logit; squash to (0,1)
    return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)
