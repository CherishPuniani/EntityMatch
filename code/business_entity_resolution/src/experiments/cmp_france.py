"""Label-free France comparison of two tree score files (raw p2) and two corrected submissions (p2n)."""
import polars as pl, sys
s1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
fr = s1.filter(pl.col("country") == "France").select("s1"); n = len(fr)
for name, tree, sub in [("original", "work/test_scores_dense_s1.parquet", "output_shift_routed/scores.parquet"),
                        ("frnorm2", "work2/test_scores_tree_frnorm2.parquet", "output_shift_frnorm2/scores.parquet")]:
    t = pl.read_parquet(tree, columns=["s1", "t", "p2"]).join(fr, on="s1")
    hi = (t["p2"] >= 0.95).sum() * 1000 / n; mid = ((t["p2"] >= 0.5) & (t["p2"] < 0.95)).sum() * 1000 / n
    lo = ((t["p2"] >= 0.1) & (t["p2"] < 0.5)).sum() * 1000 / n
    s = pl.read_parquet(sub, columns=["s1", "t", "p2n"]).join(fr, on="s1")
    s = s.with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
    g = s.group_by("s1").agg(((pl.col("p2n") >= 0.7) & (pl.col("rt") == 1)).sum().alias("k"), pl.col("p2n").max().alias("pm"))
    ue = g.filter((pl.col("k") == 0) & (pl.col("pm") >= 0.1)).height * 1000 / n
    ce = g.filter((pl.col("k") == 0) & (pl.col("pm") < 0.1)).height * 1000 / n
    print(f"{name:9s} tree p2 per 1k France S1: hi {hi:.1f} mid {mid:.1f} low(0.1-0.5) {lo:.1f} | submission: matches/S1 {g['k'].mean():.3f} "
          f"empty {(g['k']==0).mean()*100:.2f}% (uncertain-empty {ue:.1f}/1k, confident-empty {ce:.1f}/1k)")
print("reference US/India (test, submitted): uncertain-empty 2.5-3.5/1k, confident-empty 55-56/1k, matches/S1 3.36-3.38; train hi+mid ≈ 3380-3388/1k")
