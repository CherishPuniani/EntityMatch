"""Label-free test readout comparing two scored test sets (no test labels exist): per-country predicted-singleton rate,
mean matches per S1, match overlap, and (optionally) the CE score distribution per country on gated pairs.
usage: test_readout.py <test_scores_base.parquet> <test_scores_new.parquet> [ce_test.parquet] [th=0.7]"""
import sys, polars as pl

TH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
s1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"), "country")


def sel(p):
    d = pl.read_parquet(p, columns=["s1", "t", "p2"])
    return d.with_columns(pl.col("p2").rank("ordinal", descending=True).over("t").alias("rt")).filter(
        (pl.col("p2") >= TH) & (pl.col("rt") == 1)).select("s1", "t")


A, B = sel(sys.argv[1]), sel(sys.argv[2])
for name, M in (("base", A), ("new", B)):
    k = s1.join(M.group_by("s1").len("k"), on="s1", how="left").with_columns(pl.col("k").fill_null(0))
    print(name, k.group_by("country").agg((pl.col("k") == 0).mean().round(4).alias("pred_singleton"),
                                          pl.col("k").mean().round(3).alias("matches_per_s1")).sort("country").rows())
inter = len(A.join(B, on=["s1", "t"]))
print(f"matches base {len(A)} new {len(B)} common {inter} jaccard {inter / (len(A) + len(B) - inter):.4f}")
diff = pl.concat([A.join(B, on=["s1", "t"], how="anti").with_columns(pl.lit("removed").alias("chg")),
                  B.join(A, on=["s1", "t"], how="anti").with_columns(pl.lit("added").alias("chg"))]).join(s1, on="s1")
print("changed matches per 1000 S1 by country:")
print(diff.group_by("country", "chg").len().join(s1.group_by("country").len("n"), on="country")
      .with_columns((1000 * pl.col("len") / pl.col("n")).round(2).alias("per1000")).sort("country", "chg"))
if len(sys.argv) > 3 and sys.argv[3] != "-":
    ce = pl.read_parquet(sys.argv[3]).join(s1, on="s1")
    print("CE score (logit) on gated test pairs by country:")
    print(ce.group_by("country").agg(pl.len(), pl.col("ce").mean().round(3).alias("mean"),
                                     (pl.col("ce") > 0).mean().round(4).alias("frac_pos"),
                                     pl.col("ce").quantile(0.1).round(2).alias("q10"),
                                     pl.col("ce").quantile(0.5).round(2).alias("q50"),
                                     pl.col("ce").quantile(0.9).round(2).alias("q90")).sort("country"))
