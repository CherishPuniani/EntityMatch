"""France fr5: copy-count of ACCEPTED links by class x confidence band (other accepted links of the S1 excluding the link).
Reference from the LB-confirmed fr5 change: copy-like 3.07, sibling-like 3.25."""
import polars as pl, glob
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
fr = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
sc = pl.read_parquet(f"{SP}/fr5/output_fr5/scores.parquet", columns=["s1", "t", "p2n"]).join(fr, on="s1").with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
acc = sc.filter((pl.col("p2n") >= 0.7) & (pl.col("rt") == 1))
FC = ["s1", "t", "hn_rel", "hn_eq", "aempty_2", "cs_tset", "acs_tset"]
f = pl.concat([pl.read_parquet(p, columns=FC).join(acc.select("s1", "t"), on=["s1", "t"], how="semi") for d in [f"{SP}/fr5/work_unseen/feat_testun", f"{SP}/fr5/work_unseen/feat_testun_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
x = acc.join(f, on=["s1", "t"]).with_columns(pl.concat_str([
    pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq")).when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum")).otherwise(pl.lit("othnum")),
    pl.when(pl.col("cs_tset") >= 99.9).then(pl.lit("n=")).when(pl.col("cs_tset") >= 80).then(pl.lit("n~")).otherwise(pl.lit("n≠"))], separator="|").alias("c"),
    pl.col("p2n").cut([0.9, 0.99, 0.999]).cast(pl.Utf8).alias("band"))
k = acc.group_by("s1").len()
x = x.join(k, on="s1").with_columns((pl.col("len") - 1).alias("other"))
g = x.group_by("c", "band").agg(pl.len().alias("links"), pl.col("other").mean().round(3).alias("mean_other")).filter(pl.col("links") >= 500)
print(g.sort("c", "band"))
