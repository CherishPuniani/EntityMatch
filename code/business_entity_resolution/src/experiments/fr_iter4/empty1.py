"""Train (labels): empty-address targets. (1) true copies per S1 distribution (overall, per source);
(2) for empty-address targets with exact-name candidates: k = #exact-name candidate S1s, truth location;
(3) signals for the right S1: other true copies of the S1 in the same source, total true copies."""
import polars as pl, glob
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
o = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "y", "p2", "fold"])
keep = pl.read_parquet(f"{R}/work/keep_trainD.parquet").rename({"id": "s1"})
gt = o.filter(pl.col("y") == 1).with_columns((pl.col("t") // 10_000_000_000).alias("src"))
per = keep.join(gt.group_by("s1").agg(pl.len().alias("n"), (pl.col("src") == 2).sum().alias("n2"), (pl.col("src") == 3).sum().alias("n3")), on="s1", how="left").fill_null(0)
print("true in-candidate copies per S1:", per["n"].value_counts().sort("n").rows()[:12])
print("S2 copies per S1:", per["n2"].value_counts().sort("n2").rows()[:8], "| S3:", per["n3"].value_counts().sort("n3").rows()[:8])
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "aempty_2", "cs_tset", "hn_eq"]) for d in ["work/feat_trainD", "work/feat_trainD_dense"] for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))])
x = o.join(f, on=["s1", "t"]).filter((pl.col("aempty_2") == 1) & (pl.col("cs_tset") >= 99.9))
x = x.with_columns(pl.len().over("t").alias("k"), pl.col("y").sum().over("t").alias("has_true"))
tg = x.group_by("t").agg(pl.col("k").first(), pl.col("has_true").first(), pl.col("p2").max().alias("pmax"), (pl.col("p2") >= 0.7).any().alias("acc"))
print("empty-address exact-name targets:", tg.height, "| share with a true S1 among them:", round(tg["has_true"].mean(), 3))
print(tg.group_by("k").agg(pl.len(), pl.col("has_true").mean().round(3).alias("true_S1_present"), pl.col("acc").mean().round(3).alias("accepted")).sort("k").head(10))
# signals: for targets with a true S1 and k>=2
x2 = x.filter((pl.col("has_true") == 1) & (pl.col("k") >= 2)).with_columns((pl.col("t") // 10_000_000_000).alias("src"))
x2 = x2.join(per.select("s1", "n", "n2", "n3"), on="s1", how="left").with_columns(
    (pl.col("n") - pl.col("y")).alias("other_true"),
    (pl.when(pl.col("src") == 2).then(pl.col("n2")).otherwise(pl.col("n3")) - pl.col("y")).alias("other_true_same_src"))
print("k>=2 targets with a true S1:", x2["t"].n_unique())
print(x2.group_by("y").agg(pl.len(), pl.col("other_true").mean().round(3), pl.col("other_true_same_src").mean().round(3), pl.col("p2").mean().round(3)))
print("true S1 has the fewest other true copies among the target's exact-name S1s:",
      round(x2.with_columns((pl.col("other_true") == pl.col("other_true").min().over("t")).alias("ismin")).filter(pl.col("y") == 1)["ismin"].mean(), 3))
