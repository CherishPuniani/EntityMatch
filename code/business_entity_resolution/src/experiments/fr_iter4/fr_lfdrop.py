"""France, same address (hn_eq, acs>=90), identical core name: acceptance by (S1 legal form, target legal form)."""
import polars as pl, glob, sys
sys.path.insert(0, "/home2/home/amritanshu_t/amazon-mlc-26/run/work2/src")
from lfutil import lf
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(200)
sc = pl.read_parquet(f"{R}/output_final_v3/scores.parquet", columns=["s1", "t", "p2n"])
tree = pl.read_parquet(f"{R}/work2/test_scores_tree_unseen.parquet", columns=["s1", "t", "p2"])
recs = pl.concat([pl.read_parquet(f"{R}/work_unseen/p_test_s{s}.parquet", columns=["id", "country", "core"]).filter(pl.col("country") == "France").select(
    (pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), pl.col("core").list.sort().list.join(" ").alias("cs")) for s in (1, 2, 3)])
raw = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), lf("business_name").alias("lf")) for s in (1, 2, 3)])
recs = recs.join(raw, on="id")
x = tree.join(recs.rename({"id": "s1", "cs": "cs1", "lf": "lf1"}), on="s1").join(recs.rename({"id": "t", "cs": "cs2", "lf": "lf2"}), on="t")
x = x.filter(pl.col("cs1") == pl.col("cs2"))
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_eq", "aempty_2", "acs_tset"]).join(x.select("s1", "t"), on=["s1", "t"], how="semi")
               for d in ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"] for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))])
x = x.join(f, on=["s1", "t"]).filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1) & (pl.col("acs_tset") >= 90)).join(sc, on=["s1", "t"])
x = x.with_columns(pl.when(pl.col("lf1") == "").then(pl.lit("-")).otherwise(pl.col("lf1")).alias("lf1"),
                   pl.when(pl.col("lf2") == "").then(pl.lit("-")).otherwise(pl.col("lf2")).alias("lf2"))
g = x.filter((pl.col("lf1") == "-") | (pl.col("lf2") == "-") | (pl.col("lf1") == pl.col("lf2"))).group_by("lf1", "lf2").agg(
    pl.len().alias("n"), (pl.col("p2") >= 0.7).mean().round(3).alias("acc_tree"), (pl.col("p2n") >= 0.7).mean().round(3).alias("acc_final"))
print(g.filter(pl.col("n") >= 300).sort("lf1", "lf2"))
