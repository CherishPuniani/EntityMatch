"""France (fr5+veto): copy-count test for uncertain best-S1 classes (0.1<=p2n<0.7), same-address split by name class
and address-similarity; references: accepted known fillers (~3.08 copy-like) vs type-word swaps (~3.22-3.26 sibling-like)."""
import polars as pl, glob, sys
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
sys.path.insert(0, SP)
from copycount import test
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
fr = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
sc = pl.read_parquet(f"{SP}/fr5/output_fr5/scores.parquet", columns=["s1", "t", "p2n"]).join(fr, on="s1").with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
acc = sc.filter((pl.col("p2n") >= 0.7) & (pl.col("rt") == 1)).select("s1", "t")
u = sc.filter((pl.col("p2n") >= 0.1) & (pl.col("p2n") < 0.7) & (pl.col("rt") == 1))
FC = ["s1", "t", "hn_eq", "hn_rel", "aempty_2", "acs_tset", "cs_tset"]
f = pl.concat([pl.read_parquet(p, columns=FC).join(u.select("s1", "t"), on=["s1", "t"], how="semi") for d in [f"{SP}/fr5/work_unseen/feat_testun", f"{SP}/fr5/work_unseen/feat_testun_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
u = u.join(f, on=["s1", "t"]).with_columns(pl.concat_str([
    pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq")).when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum")).otherwise(pl.lit("othnum")),
    pl.when(pl.col("cs_tset") >= 99.9).then(pl.lit("n=")).when(pl.col("cs_tset") >= 80).then(pl.lit("n~")).otherwise(pl.lit("n≠")),
    pl.when(pl.col("aempty_2") == 1).then(pl.lit("")).when(pl.col("acs_tset") >= 90).then(pl.lit("a=")).otherwise(pl.lit("a≠"))], separator="|").alias("add1"))
print(u.group_by("add1").agg(pl.len().alias("n"), pl.col("p2n").mean().round(3).alias("mean_p")).sort("n", descending=True).head(16))
cl = [c for c, n in u.group_by("add1").len().iter_rows() if n >= 1000]
test(u, acc, {c: [c] for c in sorted(cl)}, "France fr5: uncertain best-S1 classes (other = accepted links)")
