import polars as pl, glob, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
from usin_gap import cls, R
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
FC = ["s1", "t", "hn_rel", "hn_eq", "aempty_2", "cs_tset"]
fr = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
sc = pl.read_parquet(f"{SP}/fr5/output_fr5/scores.parquet", columns=["s1", "t", "p2n"]).join(fr, on="s1").with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
sc = sc.filter(pl.col("p2n") >= 0.1)
f = pl.concat([pl.read_parquet(p, columns=FC).join(sc.select("s1", "t"), on=["s1", "t"], how="semi") for d in [f"{SP}/fr5/work_unseen/feat_testun", f"{SP}/fr5/work_unseen/feat_testun_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
x = cls(sc.join(f, on=["s1", "t"]))
N = fr.height
g = x.group_by("c").agg(((pl.col("p2n") >= 0.7) & (pl.col("rt") == 1)).sum().alias("acc"), ((pl.col("p2n") < 0.7) & (pl.col("rt") == 1)).sum().alias("uncert_best"))
g = g.with_columns((pl.col("acc") * 1000 / N).round(1).alias("acc_per1k"), (pl.col("uncert_best") * 1000 / N).round(1).alias("uncert_per1k")).sort("c")
pl.Config.set_tbl_rows(40)
print(g)
