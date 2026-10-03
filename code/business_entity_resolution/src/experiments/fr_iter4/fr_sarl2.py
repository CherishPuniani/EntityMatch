"""Which features differ between 'S1 without LF + copy adds SARL' and '+ copy adds SAS/SCI/EURL/SA/SASU' (France)?"""
import polars as pl, glob, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
sys.path.insert(0, "/home2/home/amritanshu_t/amazon-mlc-26/run/work2/src")
import filler_train as ft
from lfutil import lf
R = ft.R
x = ft.load("test", "France", f"{R}/output_final_v3/scores.parquet", ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], "p2n")
x = x.filter(pl.col("add1").is_in(["sa", "sci", "sas", "sasu", "eurl", "sarl"]) & ~pl.col("drop"))
recs = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), lf("business_name").alias("lf")) for s in (1, 2, 3)])
x = x.join(recs.select(pl.col("id").alias("s1"), pl.col("lf").alias("lf1")), on="s1").filter(pl.col("lf1") == "").select("s1", "t", "add1")
F = pl.concat([pl.read_parquet(p).join(x.select("s1", "t"), on=["s1", "t"], how="semi") for d in ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"]
               for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))], how="diagonal_relaxed")
d = pl.read_parquet(f"{R}/work/dense/pairfeats_test.parquet").join(x.select("s1", "t"), on=["s1", "t"], how="semi")
F = F.join(d, on=["s1", "t"], how="left").join(x, on=["s1", "t"])
F = F.with_columns((pl.col("add1") == "sarl").alias("is_sarl"))
cols = [c for c in F.columns if c not in ("s1", "t", "add1", "is_sarl") and F[c].dtype.is_numeric()]
rows = []
for c in cols:
    a = F.filter(pl.col("is_sarl"))[c].cast(pl.Float64); b = F.filter(~pl.col("is_sarl"))[c].cast(pl.Float64)
    sd = F[c].cast(pl.Float64).std() or 1.0
    rows.append((c, a.mean(), b.mean(), (a.mean() - b.mean()) / sd if sd else 0.0))
t = pl.DataFrame(rows, schema=["feature", "mean_sarl", "mean_other", "std_diff"], orient="row").with_columns(pl.col("std_diff").abs().alias("abs")).sort("abs", descending=True)
pl.Config.set_tbl_rows(25); pl.Config.set_tbl_width_chars(200)
print(t.head(20))
