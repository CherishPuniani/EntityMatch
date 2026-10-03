"""Model-implied expected F0.5 per country for a scored submission (p2n after correction), and structure of the
predicted-empty S1s. Train OOF of the same model gives the estimator's bias (expected vs actual)."""
import polars as pl, sys
TH = 0.7
def expf(df, col):
    df = df.with_columns(pl.col(col).rank("ordinal", descending=True).over("t").alias("rt"))
    df = df.with_columns(((pl.col(col) >= TH) & (pl.col("rt") == 1)).alias("sel"))
    return df.group_by("s1").agg(pl.col("sel").sum().alias("k"), pl.col(col).filter(pl.col("sel")).sum().alias("etp"),
        pl.col(col).sum().alias("eg"), (1 - pl.col(col).clip(0, 0.999999)).log().sum().alias("lp0"), pl.col(col).max().alias("pmax")
    ).with_columns(pl.when(pl.col("k") == 0).then(pl.col("lp0").exp()).otherwise(5 * pl.col("etp") / (pl.col("eg") + 4 * pl.col("k"))).alias("ef"))
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
s1tr = pl.read_parquet("work/p_train_s1.parquet", columns=["id", "country"]).with_columns((pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
keep = pl.read_parquet("work/keep_trainD.parquet").rename({"id": "s1"})
o = pl.read_parquet("work2/oof_ce3.parquet", columns=["s1", "t", "y", "p2"])
a = keep.join(s1tr, on="s1").join(expf(o, "p2"), on="s1", how="left").fill_null(0)
print("train OOF ce3 (expected F by country):", a.group_by("country").agg(pl.col("ef").mean()).sort("country").rows())
sc = pl.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "output_shift_routed/scores.parquet")
b = s1te.join(expf(sc, "p2n"), on="s1", how="left").fill_null(0)
print("test (p2n) expected F by country:", b.group_by("country").agg(pl.col("ef").mean(), (pl.col("k") == 0).mean().alias("empty"), pl.col("k").mean().alias("k"), pl.col("eg").mean().alias("EG")).sort("country").rows())
b = b.with_columns(pl.when(pl.col("pmax") >= 0.7).then(pl.lit("a>=0.7")).when(pl.col("pmax") >= 0.5).then(pl.lit("b0.5-0.7")).when(pl.col("pmax") >= 0.3).then(pl.lit("c0.3-0.5")).when(pl.col("pmax") >= 0.1).then(pl.lit("d0.1-0.3")).otherwise(pl.lit("e<0.1")).alias("pm"))
print("predicted-empty S1 per 1k, by top-candidate p2n:")
e = b.filter(pl.col("k") == 0).group_by("country", "pm").len()
n = dict(s1te.group_by("country").len().iter_rows())
print(e.with_columns((pl.col("len") * 1000 / pl.col("country").replace_strict(n)).alias("per1k")).pivot(on="country", index="pm", values="per1k").sort("pm"))
a = a.with_columns(pl.when(pl.col("pmax") >= 0.7).then(pl.lit("a>=0.7")).when(pl.col("pmax") >= 0.5).then(pl.lit("b0.5-0.7")).when(pl.col("pmax") >= 0.3).then(pl.lit("c0.3-0.5")).when(pl.col("pmax") >= 0.1).then(pl.lit("d0.1-0.3")).otherwise(pl.lit("e<0.1")).alias("pm"))
ntr = dict(keep.join(s1tr, on="s1").group_by("country").len().iter_rows())
print("train OOF predicted-empty per 1k:")
print(a.filter(pl.col("k") == 0).group_by("country", "pm").len().with_columns((pl.col("len") * 1000 / pl.col("country").replace_strict(ntr)).alias("per1k")).pivot(on="country", index="pm", values="per1k").sort("pm"))
