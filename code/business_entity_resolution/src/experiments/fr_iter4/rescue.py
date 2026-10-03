"""Empty-S1 rescue: S1 with no accepted link whose best eligible candidate has LO <= p < 0.7.
Train OOF (labels): true rate per p band (break-even for a zero-link S1 is 0.5). Test: counts per country."""
import polars as pl
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
pl.Config.set_tbl_rows(40)


def rescue(sc):
    sc = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
    has = sc.filter((pl.col("p") >= 0.7) & (pl.col("rt") == 1)).select("s1").unique()
    el = sc.filter(pl.col("rt") == 1).join(has, on="s1", how="anti").sort("p", descending=True).group_by("s1").first()
    return el.filter((pl.col("p") >= 0.4) & (pl.col("p") < 0.7)).with_columns(pl.col("p").cut([0.5, 0.6]).cast(pl.Utf8).alias("band"))


o = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "y", "p2"]).rename({"p2": "p"})
r = rescue(o)
print("OOF US/India: zero-link S1s by best-candidate band:", r.group_by("band").agg(pl.len(), pl.col("y").mean().round(3).alias("true")).sort("band").rows(), "| of 1.8M S1")
sc = pl.read_parquet(f"{SP}/fr6/output_fr6a/scores.parquet", columns=["s1", "t", "p2n"]).rename({"p2n": "p"})
c = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"), "country")
r = rescue(sc).join(c, on="s1")
print("TEST:", r.group_by("country", "band").len().sort("country", "band").rows())
