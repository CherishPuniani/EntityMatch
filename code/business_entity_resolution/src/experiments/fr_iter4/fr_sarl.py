"""Same-address pairs adding one legal form (France): acceptance split by the S1's own legal form, and the tree p2
before correction/veto; US/India train analogue (add inc/llc/ltd to S1 without LF)."""
import polars as pl, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
sys.path.insert(0, "/home2/home/amritanshu_t/amazon-mlc-26/run/work2/src")
import filler_train as ft
from lfutil import lf
R = ft.R
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
x = ft.load("test", "France", f"{R}/output_final_v3/scores.parquet", ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], "p2n")
LFS = ["sa", "sci", "sas", "sasu", "eurl", "sarl"]
x = x.filter(pl.col("add1").is_in(LFS) & ~pl.col("drop"))
recs = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "business_name", "business_address", lf("business_name").alias("lf")) for s in (1, 2, 3)])
x = x.join(recs.select(pl.col("id").alias("s1"), pl.col("lf").alias("lf1")), on="s1").join(recs.select(pl.col("id").alias("t"), pl.col("lf").alias("lf2")), on="t")
tree = pl.read_parquet(f"{R}/work2/test_scores_tree_unseen.parquet", columns=["s1", "t", "p2"]).rename({"p2": "ptree"})
x = x.join(tree, on=["s1", "t"], how="left").with_columns((pl.col("lf1") == "").alias("s1_noLF"))
print(x.group_by("add1", "s1_noLF").agg(pl.len().alias("n"), (pl.col("p") >= 0.7).mean().round(3).alias("acc"), (pl.col("ptree") >= 0.7).mean().round(3).alias("acc_tree"),
                                        pl.col("ptree").mean().round(3).alias("mean_ptree")).sort("add1", "s1_noLF"))
info = {r[0]: (r[1], r[2]) for r in recs.iter_rows()}
for s1, t, p, pt in x.filter((pl.col("add1") == "sarl") & pl.col("s1_noLF") & (pl.col("p") < 0.7)).sample(8, seed=1).select("s1", "t", "p", "ptree").iter_rows():
    a, b = info[s1], info[t]
    print(f"p2n={p:.3f} tree={pt:.3f}  S1 {a[0][:34]:34s} | {a[1][:55]}\n{'':24s}T  {b[0][:34]:34s} | {b[1][:55]}")
