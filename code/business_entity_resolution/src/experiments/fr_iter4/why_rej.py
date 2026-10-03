"""fr5+veto: why are 15-20 % of same-address filler pairs (groupe/developpement/france, and reference services/cie) still rejected?"""
import polars as pl, sys
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
sys.path.insert(0, SP); sys.path.insert(0, "/home2/home/amritanshu_t/amazon-mlc-26/run/work2/src")
import filler_train as ft
from lfutil import lf
R = ft.R
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
x = ft.load("test", "France", f"{SP}/fr5/output_fr5/scores.parquet", ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], "p2n")
x = x.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
tree = pl.read_parquet(f"{SP}/fr5/test_scores_tree_unseen_fr5.parquet", columns=["s1", "t", "p2"]).rename({"p2": "ptree"})
raw = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), lf("business_name").alias("lf")) for s in (1, 2, 3)])
x = x.join(tree, on=["s1", "t"], how="left").join(raw.rename({"id": "s1", "lf": "lf1"}), on="s1").join(raw.rename({"id": "t", "lf": "lf2"}), on="t")
x = x.filter(pl.col("add1").is_in(["groupe", "developpement", "france", "services", "cie"]))
x = x.with_columns(pl.when((pl.col("p") >= 0.7) & (pl.col("rt") == 1)).then(pl.lit("accepted"))
                   .when((pl.col("lf1") != "") & (pl.col("lf2") != "") & (pl.col("lf1") != pl.col("lf2"))).then(pl.lit("rej: legal-form change"))
                   .when(pl.col("rt") > 1).then(pl.lit("rej: target claimed by another S1"))
                   .when(pl.col("ptree") >= 0.7).then(pl.lit("rej: correction (tree>=0.7)"))
                   .otherwise(pl.lit("rej: tree < 0.7")).alias("why"))
print(x.group_by("add1", "why").len().pivot(on="add1", index="why", values="len").fill_null(0).sort("why"))
