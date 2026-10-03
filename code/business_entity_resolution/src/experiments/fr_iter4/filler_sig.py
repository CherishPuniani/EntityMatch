"""Per added token (same number+street, exactly one added token): n with / without a dropped S1 content token,
replace ratio, cluster rate (another candidate of the same S1 adds the same token), true rate (train) / acceptance (test).
Question: does the replace ratio separate copy fillers (true ~1) from sibling descriptors (true ~0) in labelled data?"""
import polars as pl, glob, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
from filler_train import load, R
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(220)


def sig(x, lab, has_y, nmin):
    x = x.filter(~pl.col("add1").is_in(["none", "2+"]))
    x = x.with_columns((pl.len().over("s1", "add1") > 1).alias("clu"))
    agg = [pl.len().alias("n"), (~pl.col("drop")).sum().alias("n_add"), pl.col("drop").sum().alias("n_repl"),
           pl.col("clu").mean().round(3).alias("cluster"), (pl.col("p") >= 0.7).mean().round(3).alias("acc"),
           (pl.col("p").filter(~pl.col("drop")) >= 0.7).mean().round(3).alias("acc_add"),
           (pl.col("p").filter(pl.col("drop")) >= 0.7).mean().round(3).alias("acc_repl")]
    if has_y:
        agg += [pl.col("y").mean().round(3).alias("true"), pl.col("y").filter(~pl.col("drop")).mean().round(3).alias("true_add"),
                pl.col("y").filter(pl.col("drop")).mean().round(3).alias("true_repl")]
    g = x.group_by("add1").agg(agg).filter(pl.col("n") >= nmin).with_columns((pl.col("n_repl") / pl.col("n_add")).round(2).alias("repl_ratio"))
    print(f"\n== {lab}"); print(g.sort("n", descending=True))
    if has_y:
        print(g.with_columns(pl.col("true").cut([0.2, 0.8]).alias("cls")).group_by("cls").agg(
            pl.len().alias("tokens"), pl.col("repl_ratio").median().alias("median_repl_ratio"), pl.col("repl_ratio").min().alias("min"),
            pl.col("repl_ratio").max().alias("max"), pl.col("cluster").median().alias("median_cluster")).sort("cls"))


if __name__ == "__main__":
    import filler_train as ft
    for c in ("India", "US"):
        sig(ft.load("train", c, f"{R}/work2/oof_ce4q.parquet", ["work/feat_trainD", "work/feat_trainD_dense"], "p2"), f"TRAIN {c}", True, 400)
    sig(ft.load("test", "France", f"{R}/output_final_v3/scores.parquet", ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], "p2n"),
        "TEST France", False, 300)
