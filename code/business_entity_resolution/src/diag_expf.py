"""Label-free diagnostic: model-implied expected F0.5 per S1 (calibration assumed), per country.
Train OOF (labels known) is used to check the estimator against the actual F."""
import polars as pl, numpy as np, sys
TH = 0.7

def expf(df):
    df = df.with_columns(pl.col("p2").rank("ordinal", descending=True).over("t").alias("rt"))
    df = df.with_columns(((pl.col("p2") >= TH) & (pl.col("rt") == 1)).alias("sel"))
    agg = df.group_by("s1").agg(
        pl.col("sel").sum().alias("k"),
        pl.col("p2").filter(pl.col("sel")).sum().alias("etp"),
        pl.col("p2").sum().alias("eg"),
        (1 - pl.col("p2")).log().sum().alias("lp0"),
        ((pl.col("p2") > 0.1) & (pl.col("p2") < 0.9)).sum().alias("nunc"),
        pl.col("p2").max().alias("pmax"))
    agg = agg.with_columns(pl.when(pl.col("k") == 0).then(pl.col("lp0").exp())
                           .otherwise(5 * pl.col("etp") / (pl.col("eg") + 4 * pl.col("k"))).alias("ef"))
    return agg

s1tr = pl.read_parquet("work/p_train_s1.parquet", columns=["id", "country"]).with_columns(
    (pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
keep = pl.read_parquet("work/keep_trainD.parquet").rename({"id": "s1"})
gt = pl.read_parquet("work/gt_pairs_int.parquet")

for name, path in [("oof_final", "work/oof_dense_s1_ce_qwen.parquet"), ("oof_dense_s1", "work/oof_dense_s1.parquet")]:
    o = pl.read_parquet(path, columns=["s1", "t", "y", "p2"])
    a = expf(o)
    # actual F
    sel = o.with_columns(pl.col("p2").rank("ordinal", descending=True).over("t").alias("rt")).filter((pl.col("p2") >= TH) & (pl.col("rt") == 1))
    tp = sel.group_by("s1").agg(pl.len().alias("k2"), pl.col("y").sum().alias("tp"))
    G = gt.group_by("s1").agg(pl.len().alias("G"))
    d = keep.join(s1tr, on="s1").join(a, on="s1", how="left").join(tp, on="s1", how="left").join(G, on="s1", how="left").fill_null(0)
    d = d.with_columns(pl.when(pl.col("G") == 0).then((pl.col("k2") == 0).cast(pl.Float64)).when(pl.col("k2") == 0).then(0.0)
                       .otherwise(5 * pl.col("tp") / (pl.col("G") + 4 * pl.col("k2"))).alias("f"))
    print(name)
    print(d.group_by("country").agg(pl.len(), pl.col("f").mean().alias("actualF"), pl.col("ef").mean().alias("expF"),
          (pl.col("k") == 0).mean().alias("pred_empty"), pl.col("k").mean().alias("k"), pl.col("eg").mean().alias("EG"),
          (pl.col("nunc") > 0).mean().alias("frac_unc_s1"), pl.col("nunc").mean().alias("nunc")).sort("country"))

for name, path in [("test_routed", "work/test_scores_final_routed.parquet"), ("test_dense_s1", "work/test_scores_dense_s1.parquet"),
                   ("test_ce_all", "work/test_scores_dense_s1_ce_qwen.parquet")]:
    t = pl.read_parquet(path, columns=["s1", "t", "p2"])
    a = s1te.join(expf(t), on="s1", how="left").fill_null(0)
    print(name)
    print(a.group_by("country").agg(pl.len(), pl.col("ef").mean().alias("expF"), (pl.col("k") == 0).mean().alias("pred_empty"),
          pl.col("k").mean().alias("k"), pl.col("eg").mean().alias("EG"), (pl.col("nunc") > 0).mean().alias("frac_unc_s1"),
          pl.col("nunc").mean().alias("nunc")).sort("country"))
    print("overall expF", a["ef"].mean())
