"""(1) Accepted links per 1k S1 by address stratum: France (v3) vs US/India test (v3) vs US/India train truth.
(2) Empty-address targets with the S1's exact core name: acceptance, and copy-count test (France)."""
import polars as pl, glob, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
from copycount import test
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
FC = ["s1", "t", "hn_rel", "hn_eq", "aempty_2", "hn_any"]


def strat(df):
    return df.with_columns(pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq"))
                           .when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum"))
                           .otherwise(pl.concat_str(pl.lit("r"), pl.col("hn_rel").cast(pl.Int32).cast(pl.Utf8))).alias("st"))


def feats(dirs, keys):
    return pl.concat([pl.read_parquet(p, columns=FC).join(keys, on=["s1", "t"], how="semi") for d in dirs for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))])


m = pl.read_csv(f"{R}/output_final_v3/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
links = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != "").select(
             (pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"),
             (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000)
              + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t")))
c = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"), "country")
nS1 = dict(c.group_by("country").len().iter_rows())
l = strat(links.join(c, on="s1").join(feats(["work/feat_test", "work/feat_test_dense"], links), on=["s1", "t"], how="left"))
g = l.group_by("country", "st").len().with_columns((pl.col("len") * 1000 / pl.col("country").replace_strict(nS1)).round(1).alias("per1k"))
print("TEST accepted links per 1k S1 by stratum (v3)")
print(g.pivot(on="country", index="st", values="per1k").sort("st"))
# train truth
o = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "y"]).filter(pl.col("y") == 1)
ctr = pl.read_parquet(f"{R}/work/p_train_s1.parquet", columns=["id", "country"]).select((pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"), "country")
keep = pl.read_parquet(f"{R}/work/keep_trainD.parquet").rename({"id": "s1"})
ntr = dict(keep.join(ctr, on="s1").group_by("country").len().iter_rows())
lt = strat(o.join(ctr, on="s1").join(feats(["work/feat_trainD", "work/feat_trainD_dense"], o.select("s1", "t")), on=["s1", "t"], how="left"))
gt = lt.group_by("country", "st").len().with_columns((pl.col("len") * 1000 / pl.col("country").replace_strict(ntr)).round(1).alias("per1k"))
print("TRAIN true links (in candidates) per 1k S1 by stratum")
print(gt.pivot(on="country", index="st", values="per1k").sort("st"))
