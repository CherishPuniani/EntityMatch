"""Shared feature assembly for stage 1 / stage 2 (train and test)."""
import polars as pl, numpy as np, glob

ATTR = ["hn_h", "cc_h", "acs_h", "nums_h", "cc_fs1", "acs_fs1", "cc_ft", "acs_ft"]


def load_feats(featdir, split, attrs=None):
    """featdir may list several comma-separated dirs (e.g. blocking + dense-channel pairs); EXTRA_PAIRFEATS names an
    optional (s1, t, ...) table of extra pair features, left-joined (null where absent)."""
    import os
    f = pl.concat([pl.read_parquet(p) for d in featdir.split(",") for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
    if os.environ.get("EXTRA_PAIRFEATS"):
        f = f.join(pl.read_parquet(os.environ["EXTRA_PAIRFEATS"]), on=["s1", "t"], how="left", maintain_order="left")
    at = pl.read_parquet(f"work/rec_attrs_{attrs or split}.parquet")
    f = f.join(at.rename({c: c + "_s" for c in ATTR}), left_on="s1", right_on="id", how="left", maintain_order="left")
    f = f.join(at.rename({c: c + "_t" for c in ATTR}), left_on="t", right_on="id", how="left", maintain_order="left")
    return f


def stage1_context(f):
    """cheap group features that don't need model scores."""
    f = f.with_columns(
        pl.len().over("s1").cast(pl.Float32).alias("s1_ncand"),
        (pl.col("bscore") / pl.col("bscore").max().over("s1")).alias("b_s1rel"),
        (pl.col("t_best") - pl.col("rs")).alias("t_gap"),
        (pl.col("t_bbest") - pl.col("bscore")).alias("t_bgap"),
        (pl.col("rs") / pl.col("rs").max().over("s1")).alias("rs_s1rel"),
        (pl.col("hn_h_t") == pl.col("hn_h_s")).cast(pl.Int8).alias("hn_hash_eq"),
        (pl.len().over("s1", "hn_h_t") - 1).cast(pl.Float32).alias("cnt_same_hn"),
        (pl.len().over("s1", "cc_h_t") - 1).cast(pl.Float32).alias("cnt_same_cc"),
        (pl.len().over("s1", "acs_h_t") - 1).cast(pl.Float32).alias("cnt_same_acs"),
        (pl.len().over("s1", "nums_h_t") - 1).cast(pl.Float32).alias("cnt_same_nums"),
        ((pl.col("hn_h_t") == pl.col("hn_h_s")).cast(pl.Float32).sum().over("s1")).alias("s1_hn_support"),
        ((pl.col("cc_h_t") == pl.col("cc_h_s")).cast(pl.Float32).sum().over("s1")).alias("s1_cc_support"),
    )
    return f


def stage2_context(f, p="p1"):
    f = f.with_columns(
        pl.col(p).rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("p_rank_s1"),
        pl.col(p).max().over("s1").alias("p_max_s1"),
        pl.col(p).sum().over("s1").alias("p_sum_s1"),
        (pl.col(p) >= 0.5).cast(pl.Float32).sum().over("s1").alias("p_n50_s1"),
        pl.col(p).rank("ordinal", descending=True).over("t").cast(pl.Float32).alias("p_rank_t"),
        pl.len().over("t").cast(pl.Float32).alias("n_s1_t"),
        pl.col(p).sum().over("t").alias("p_sum_t"),
        (pl.col(p).sum().over("s1", "hn_h_t") - pl.col(p)).alias("p_same_hn"),
        (pl.col(p).sum().over("s1", "cc_h_t") - pl.col(p)).alias("p_same_cc"),
        (pl.col(p).sum().over("s1", "acs_h_t") - pl.col(p)).alias("p_same_acs"),
        ((pl.col("hn_h_t") == pl.col("hn_h_s")).cast(pl.Float32) * pl.col(p)).sum().over("s1").alias("p_s1hn_support"),
    )
    # best probability of this target under any *other* S1
    top = f.group_by("t").agg(pl.col(p).top_k(2).alias("_tk")).select(
        "t", pl.col("_tk").list.get(0).alias("_m1"), pl.col("_tk").list.get(1, null_on_oob=True).fill_null(0).alias("_m2"))
    f = f.join(top, on="t", how="left", maintain_order="left").with_columns(
        pl.when(pl.col("p_rank_t") == 1).then(pl.col("_m2")).otherwise(pl.col("_m1")).alias("p_t_other")).drop("_m1", "_m2")
    f = f.with_columns(
        (pl.col(p) - pl.col("p_t_other")).alias("p_t_margin"),
        (pl.col(p) - pl.col("p_max_s1")).alias("p_s1_gap"),
        (pl.col("p_sum_s1") - pl.col(p)).alias("p_s1_others"))
    return f


DROP = {"s1", "t", "y", "fold", "p1", "p2"} | {c + "_s" for c in ATTR[:4]} | {c + "_t" for c in ATTR[:4]}


def feat_cols(f, extra_drop=()):
    return [c for c in f.columns if c not in DROP and c not in extra_drop]
