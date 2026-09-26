"""Single source of truth for every tunable knob. Serialised next to the models so a
test run can prove it used the same feature/blocking settings as training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class Config:
    # ---------------- paths ----------------
    data_dir: str = ""                # folder containing train/ and test/
    work_dir: str = "work"            # caches, memmaps, models
    out_dir: str = "output"           # matching_results.tsv + candidate_pairs.tsv
    validator: str = ""               # optional path to utils/validate_submission.py

    # ---------------- runtime --------------
    workers: int = 0                  # 0 => os.cpu_count()
    seed: int = 42
    force: bool = False               # recompute cached stages
    keep_intermediates: bool = False  # keep large per-partition feat_*/s2_*.npy after use
                                       # (default False: delete them once consumed within a
                                       # run to bound disk use — at full scale, stage-1 (fp16)
                                       # + stage-2 feature matrices for train AND test can
                                       # exceed 40 GB combined if all kept simultaneously;
                                       # pairs_*.npz/parquet caches, which are cheap in size
                                       # and expensive to recompute, are always kept)

    # ---------------- blocking -------------
    k_key: int = 40                   # max key-based candidates per S1 record
    k_dense: int = 15                 # dense neighbors kept for every record
    k_dense_script: int = 50          # also keep non-Latin neighbors out to this rank
    name_topk: int = 2                # rarest name tokens used in keys
    addr_topk: int = 4                # rarest address tokens used in keys
    num_topk: int = 2                 # first N numbers of the address used in keys
    max_df_pair: int = 500            # drop a compound key if > this many index records share it
    max_df_name: int = 200            # stricter cap for name-only keys
    max_pairs_per_key: int = 50_000   # df_query * df_index cap (kills pathological keys)
    join_budget_rows: int = 40_000_000  # raw join rows per query chunk (RAM guard)
    bm25_k: int = 5                   # union schema-agnostic BM25 top-k into candidates
    # Also keep deeper BM25 hits when the INDEX address is empty. Residual blocking
    # misses with empty addr have median BM25 rank ~33 (0% inside top-5).
    bm25_k_empty: int = 40            # 0=off; keep empty-addr neighbors out to this rank

    # ---------------- dense -----------------------------------
    # Measured winner on the 3% slice (0.991332): keys ∪ BM25@5 ∪ e5 kNN,
    # with non-Latin neighbors kept out to rank 50.
    dense: bool = True
    # e5-small is the measured model (MIT, 118M, 384-d). Literature candidate, not yet
    # scored here: ibm-granite/granite-embedding-97m-multilingual-r2 (Apache-2.0).
    dense_model: str = "intfloat/multilingual-e5-small"
    dense_batch: int = 256            # knee on M5 Air MPS; CUDA auto-ramps higher
    dense_batch_fixed: bool = False   # True => skip the encode batch probe
    dense_max_len: int = 64
    # Adaptive per-query dense budget (opt-in). easy→k_dense_easy, empty→k_dense_empty,
    # non-Latin→k_dense_script, else k_dense. Measure on --dev-frac 0.03 before promoting.
    dense_adaptive: bool = False
    k_dense_easy: int = 5
    k_dense_empty: int = 30

    # ---------------- features ------------
    feat_chunk: int = 2_000_000       # pairs per feature chunk
    hash_features: int = 2 ** 20      # hashing-trick width for TF-IDF
    # Two-stage RapidFuzz: only pairs with brank < this (plus any dense hit) get
    # expensive fuzzy scores; others leave those cols as NaN (LightGBM-ok).
    # 0 = off (all pairs). Default 24 ≈ keep top key-ranks; measured for speed.
    feat_fuzz_brank_max: int = 24

    # ---------------- training ------------
    stage1_folds: list = field(default_factory=lambda: [0, 1, 2])
    stage2_folds: list = field(default_factory=lambda: [3, 4, 5, 6, 7])
    holdout_folds: list = field(default_factory=lambda: [8, 9])
    lgb_rounds: int = 3000
    lgb_lr: float = 0.08
    lgb_leaves: int = 191
    lgb_min_leaf: int = 150
    early_stop: int = 100
    monotone: bool = True
    max_train_rows: int = 30_000_000
    dev_frac: float = 1.0             # <1.0 => consistent entity sub-sample of TRAIN (fast iteration)
    train_countries: list = field(default_factory=list)   # [] => all (LOCO experiments)

    # ---------------- decision ------------
    decision: str = "auto"            # auto | threshold | expected_f
    # Pre-adjust calibrated p using first house number before expected_f.
    housenum_boost: float = 0.05      # add to p when first house numbers match exactly
    housenum_pen: float = 0.05        # subtract when |Δ| in 1..5 (near-copy distractors)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Settings that change the *meaning* of features; must be identical at train and test.
FEATURE_CONTRACT = ("k_key", "k_dense", "k_dense_script", "name_topk", "addr_topk", "num_topk",
                    "max_df_pair", "max_df_name", "max_pairs_per_key", "bm25_k", "bm25_k_empty",
                    "dense", "dense_model", "hash_features", "feat_fuzz_brank_max",
                    "dense_adaptive", "k_dense_easy", "k_dense_empty")
