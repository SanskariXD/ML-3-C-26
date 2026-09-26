"""Entry point.

  python src/run.py all      --data-dir <.../student_resource/dataset>   # train + test end to end
  python src/run.py train    --data-dir ...                               # prepare/block/features/fit on train
  python src/run.py predict  --data-dir ...                               # prepare/block/features/infer on test
  python src/run.py prepare|block|features --split train|test --data-dir ...

Every stage is cached under --work-dir; re-running skips finished stages (use --force).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# See dense.py for why this must be set before any `import torch` (Windows cu12x DLL
# load can otherwise fail with "paging file is too small" even with ample RAM/CUDA_MODULE_LOADING).
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# Max out BLAS / OpenMP for rapidfuzz + LightGBM on Apple Silicon / Linux.
_nthreads = max(1, (os.cpu_count() or 4) - 1)
os.environ.setdefault("OMP_NUM_THREADS", str(_nthreads))
os.environ.setdefault("MKL_NUM_THREADS", str(_nthreads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_nthreads))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(_nthreads))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_nthreads))
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config  # noqa: E402
from utils import LOG, setup_logging, timed  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Business entity resolution pipeline")
    ap.add_argument("command", choices=["all", "train", "predict", "prepare", "block",
                                        "features", "blocking-report", "train-reranker",
                                        "eval-reranker"])
    ap.add_argument("--reranker-n-pos", type=int, default=300_000)
    ap.add_argument("--reranker-neg-per-pos", type=int, default=2)
    ap.add_argument("--reranker-epochs", type=int, default=1)
    ap.add_argument("--reranker-batch-size", type=int, default=64)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--validator", default="")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="keep large per-partition feat_*/s2_*.npy caches after they are "
                    "consumed (default: delete them to bound disk use — see config.py)")
    ap.add_argument("--dense", action=argparse.BooleanOptionalAction, default=True,
                    help="multilingual embedding kNN (default on; --no-dense to disable)")
    ap.add_argument("--dense-model", default=None,
                    help="sentence-transformers model id (default multilingual-e5-small)")
    ap.add_argument("--dense-batch", type=int, default=None,
                    help="pin encode batch and skip the batch-size probe")
    ap.add_argument("--k-key", type=int, default=None)
    ap.add_argument("--k-dense", type=int, default=None)
    ap.add_argument("--k-dense-script", type=int, default=None,
                    help="also keep non-Latin dense neighbors out to this rank (default 50, 0=off)")
    ap.add_argument("--bm25-k", type=int, default=None,
                    help="union schema-agnostic BM25 top-k into blocking (default 5, 0=off)")
    ap.add_argument("--bm25-k-empty", type=int, default=None,
                    help="also keep empty-address BM25 neighbors out to this rank (0=off)")
    ap.add_argument("--max-df-pair", type=int, default=None)
    ap.add_argument("--max-df-name", type=int, default=None)
    ap.add_argument("--dev-frac", type=float, default=None,
                    help="entity-consistent train sub-sample for fast iteration (e.g. 0.1)")
    ap.add_argument("--train-countries", default="",
                    help="comma list; train only on these (leave-one-country-out studies)")
    ap.add_argument("--decision", choices=["auto", "threshold", "expected_f"], default=None)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--no-monotone", action="store_true")
    ap.add_argument("--lgb-rounds", type=int, default=None)
    ap.add_argument("--min-leaf", type=int, default=None)
    ap.add_argument("--feat-fuzz-brank-max", type=int, default=None,
                    help="RapidFuzz only for brank<N (+ dense hits); 0=all (default 24)")
    ap.add_argument("--feat-chunk", type=int, default=None,
                    help="pairs per feature chunk (raise on high-RAM boxes, e.g. 5000000)")
    ap.add_argument("--join-budget-rows", type=int, default=None,
                    help="blocking join RAM guard (raise on >=64GB RAM, e.g. 80000000)")
    ap.add_argument("--dense-adaptive", action=argparse.BooleanOptionalAction, default=None,
                    help="per-query dense k: easy/empty/non-Latin budgets (measure before promote)")
    return ap.parse_args(argv)


def build_config(a) -> Config:
    cfg = Config(data_dir=os.path.abspath(a.data_dir), work_dir=os.path.abspath(a.work_dir),
                 out_dir=os.path.abspath(a.out_dir), workers=a.workers, force=a.force,
                 dense=a.dense, keep_intermediates=a.keep_intermediates)
    if not a.validator:
        guess = os.path.join(os.path.dirname(cfg.data_dir), "utils", "validate_submission.py")
        cfg.validator = guess if os.path.isfile(guess) else ""
    else:
        cfg.validator = os.path.abspath(a.validator)
    if a.k_key is not None:
        cfg.k_key = a.k_key
    if a.k_dense is not None:
        cfg.k_dense = a.k_dense
    if a.k_dense_script is not None:
        cfg.k_dense_script = a.k_dense_script
    if a.bm25_k is not None:
        cfg.bm25_k = a.bm25_k
    if a.bm25_k_empty is not None:
        cfg.bm25_k_empty = a.bm25_k_empty
    if a.dense_model:
        cfg.dense_model = a.dense_model
    if a.dense_batch is not None:
        cfg.dense_batch = a.dense_batch
        cfg.dense_batch_fixed = True
    if a.max_df_pair is not None:
        cfg.max_df_pair = a.max_df_pair
    if a.max_df_name is not None:
        cfg.max_df_name = a.max_df_name
    if a.dev_frac is not None:
        cfg.dev_frac = a.dev_frac
    if a.train_countries:
        cfg.train_countries = [c.strip() for c in a.train_countries.split(",") if c.strip()]
    if a.decision:
        cfg.decision = a.decision
    if a.max_train_rows:
        cfg.max_train_rows = a.max_train_rows
    if a.no_monotone:
        cfg.monotone = False
    if a.lgb_rounds:
        cfg.lgb_rounds = a.lgb_rounds
    if a.min_leaf:
        cfg.lgb_min_leaf = a.min_leaf
    if a.feat_fuzz_brank_max is not None:
        cfg.feat_fuzz_brank_max = a.feat_fuzz_brank_max
    if a.feat_chunk is not None:
        cfg.feat_chunk = a.feat_chunk
    if a.join_budget_rows is not None:
        cfg.join_budget_rows = a.join_budget_rows
    if a.dense_adaptive is not None:
        cfg.dense_adaptive = a.dense_adaptive
    for sub in ("train", "test"):
        p = os.path.join(cfg.data_dir, sub)
        if not os.path.isdir(p):
            raise SystemExit(f"--data-dir must contain train/ and test/ (missing {p})")
    return cfg


def stage_prepare_block_features(cfg, split):
    from blocking import run_blocking
    from features import run_features
    from prepare import load_split, prepare
    prepare(cfg, split)
    s1, idx, parts = load_split(cfg, split)
    emb = None
    if cfg.dense:
        from dense import build_embeddings
        with timed(f"dense embeddings {split}"):
            emb = build_embeddings(cfg, split, s1, idx)
    with timed(f"blocking {split}"):
        stats = run_blocking(cfg, split, s1, idx, parts, emb)
    tot = sum(v["pairs"] for v in stats.values())
    LOG.info("blocking %s: %d pairs total, %.1f per S1", split, tot, tot / max(1, len(s1)))
    with timed(f"features {split}"):
        run_features(cfg, split, s1, idx, parts, emb)
    del s1, idx


def blocking_report(cfg):
    """Recall ceiling / reduction ratio of the cached train blocking (no model needed)."""
    import numpy as np
    from blocking import load_pairs
    from prepare import load_gt, load_split
    s1, idx, parts = load_split(cfg, "train", columns=["entity_id"])
    gs, gi = load_gt(cfg)
    n_idx = len(idx)
    gt_keys = np.unique(gs.astype(np.int64) * n_idx + gi)
    hit = tot = 0
    for ck, (s1_rows, idx_rows) in parts.items():
        P = load_pairs(cfg, "train", ck)
        key = s1_rows[P["q"]].astype(np.int64) * n_idx + idx_rows[P["i"]]
        h = int(np.isin(key, gt_keys).sum())
        rr = 1 - len(key) / max(1, len(s1_rows) * len(idx_rows))
        LOG.info("  %-10r pairs=%10d  pairs/S1=%5.1f  reduction_ratio=%.6f  true_pairs=%d",
                 ck, len(key), len(key) / max(1, len(s1_rows)), rr, h)
        hit += h
        tot += len(key)
    LOG.info("RECALL CEILING = %.4f   (%d / %d true pairs), pairs/S1 = %.1f",
             hit / max(1, len(gt_keys)), hit, len(gt_keys), tot / max(1, len(s1)))


def main(argv=None):
    a = parse_args(argv)
    cfg = build_config(a)
    os.makedirs(cfg.work_dir, exist_ok=True)
    setup_logging(os.path.join(cfg.work_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    LOG.info("config: %s", cfg.to_dict())
    t0 = time.perf_counter()
    from pipeline import enforce_contract, predict_test, train_all

    if a.command in ("prepare", "block", "features"):
        from prepare import prepare
        if a.split == "test" and os.path.exists(os.path.join(cfg.work_dir, "models", "meta.json")):
            enforce_contract(cfg)
        if a.command == "prepare":
            prepare(cfg, a.split)
        else:
            stage_prepare_block_features(cfg, a.split)
    elif a.command == "blocking-report":
        blocking_report(cfg)
    elif a.command == "train-reranker":
        from pipeline import model_dir
        from rerank import train_reranker
        from utils import load_json
        meta = load_json(os.path.join(model_dir(cfg), "meta.json"))
        for k, v in meta["contract"].items():
            setattr(cfg, k, v)
        with timed("train-reranker"):
            train_reranker(cfg, n_pos=a.reranker_n_pos, neg_per_pos=a.reranker_neg_per_pos,
                           epochs=a.reranker_epochs, batch_size=a.reranker_batch_size)
    elif a.command == "eval-reranker":
        from pipeline import model_dir
        from eval_reranker import evaluate
        from utils import load_json
        meta = load_json(os.path.join(model_dir(cfg), "meta.json"))
        for k, v in meta["contract"].items():
            setattr(cfg, k, v)
        with timed("eval-reranker"):
            evaluate(cfg)
    if a.command in ("train", "all"):
        stage_prepare_block_features(cfg, "train")
        # Torch's OpenMP pool and LightGBM's libomp deadlock on macOS if both run
        # in one process (Dataset construction waits forever on a join barrier).
        # Re-exec after the cached blocking pass so training starts with a clean pool.
        if "torch" in sys.modules:
            LOG.info("re-exec without torch so LightGBM's OpenMP does not deadlock")
            os.execv(sys.executable, [sys.executable, *sys.argv])
        with timed("train"):
            train_all(cfg)
    if a.command in ("predict", "all"):
        enforce_contract(cfg)
        cfg.dev_frac = 1.0          # test split is never sub-sampled
        stage_prepare_block_features(cfg, "test")
        with timed("predict"):
            predict_test(cfg)
    LOG.info("DONE in %.1f min", (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
