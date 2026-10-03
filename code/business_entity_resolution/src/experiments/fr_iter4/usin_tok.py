"""US/India: same-address single-added-token pairs, per token: train true rate/acceptance vs test acceptance and volume
(per 1k S1). Flags tokens whose test acceptance or volume departs from train (test-only descriptors)."""
import polars as pl, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
import filler_train as ft
R = ft.R
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
ntr = {"US": 1077843, "India": 719068}
for c in ("US", "India"):
    tr = ft.load("train", c, f"{R}/work2/oof_ce4q.parquet", ["work/feat_trainD", "work/feat_trainD_dense"], "p2")
    te = ft.load("test", c, f"{R}/output_final_v3/scores.parquet", ["work/feat_test", "work/feat_test_dense"], "p2n")
    nte = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["country"]).filter(pl.col("country") == c).height
    a = tr.filter(~pl.col("add1").is_in(["none", "2+"])).group_by("add1").agg((pl.len() * 1000 / ntr[c]).round(1).alias("tr_per1k"), pl.col("y").mean().round(3).alias("tr_true"),
                                                                              (pl.col("p") >= 0.7).mean().round(3).alias("tr_acc"))
    b = te.filter(~pl.col("add1").is_in(["none", "2+"])).group_by("add1").agg((pl.len() * 1000 / nte).round(1).alias("te_per1k"), (pl.col("p") >= 0.7).mean().round(3).alias("te_acc"),
                                                                              ((pl.col("p") >= 0.7).sum() * 1000 / nte).round(2).alias("te_acc_per1k"))
    t = b.join(a, on="add1", how="left").with_columns(((pl.col("te_acc") - pl.col("tr_true")).abs()).alias("gap"))
    print(f"\n== {c}: tokens with test accepted >= 0.3/1k S1 and |test acc - train true| >= 0.1 or unseen in train")
    print(t.filter((pl.col("te_acc_per1k") >= 0.3) & ((pl.col("gap") >= 0.1) | pl.col("tr_true").is_null())).sort("te_acc_per1k", descending=True).head(25))
