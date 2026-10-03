"""France: uncertain rejected candidates (0.1 <= p2n < 0.7, best S1 for the target) by stratum and name similarity;
copy-count test per class. Train India/US: same classes with true rate (calibration of the test)."""
import polars as pl, glob, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
from copycount import test
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
FC = ["s1", "t", "hn_rel", "hn_eq", "aempty_2", "ns_tset", "cs_tset"]


def classes(sc, fd, keys_filter):
    sc = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
    u = sc.filter((pl.col("p") >= 0.1) & (pl.col("p") < 0.7) & (pl.col("rt") == 1)).join(keys_filter, on="s1", how="semi")
    f = pl.concat([pl.read_parquet(p, columns=FC).join(u.select("s1", "t"), on=["s1", "t"], how="semi") for d in fd for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))])
    u = u.join(f, on=["s1", "t"])
    u = u.with_columns(pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq"))
                       .when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum")).otherwise(pl.lit("othernum")).alias("st"),
                       pl.when(pl.col("cs_tset") >= 99.9).then(pl.lit("name=")).when(pl.col("cs_tset") >= 80).then(pl.lit("name~")).otherwise(pl.lit("name≠")).alias("nm"))
    return u.with_columns(pl.concat_str(["st", "nm"], separator="|").alias("add1"))


# train calibration (India)
o = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "y", "p2"]).rename({"p2": "p"})
ctr = pl.read_parquet(f"{R}/work/p_train_s1.parquet", columns=["id", "country"]).select((pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"), "country")
for c in ("India", "US"):
    k = ctr.filter(pl.col("country") == c).select("s1")
    u = classes(o, ["work/feat_trainD", "work/feat_trainD_dense"], k)
    print(f"\nTRAIN {c}: uncertain rejected best-S1 candidates by class: n, true rate")
    print(u.group_by("add1").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true")).sort("n", descending=True))
    links = o.filter(pl.col("y") == 1).select("s1", "t").join(k, on="s1", how="semi")
    cl = u["add1"].unique().to_list()
    test(u, links, {x: [x] for x in cl if u.filter(pl.col("add1") == x).height >= 500}, f"TRAIN {c} copy-count (true links)")
# France
sc = pl.read_parquet(f"{R}/output_final_v3/scores.parquet", columns=["s1", "t", "p2n"]).rename({"p2n": "p"})
fr = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
u = classes(sc, ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], fr)
print("\nFRANCE: uncertain rejected best-S1 candidates by class (per 1k French S1)")
print(u.group_by("add1").agg(pl.len().alias("n"), (pl.len() * 1000 / fr.height).round(1).alias("per1k"), pl.col("p").mean().round(3).alias("mean_p")).sort("n", descending=True))
m = pl.read_csv(f"{R}/output_final_v3/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
links = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != "").select(
             (pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"),
             (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000)
              + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t"))).join(fr, on="s1", how="semi")
cl = u["add1"].unique().to_list()
test(u, links, {x: [x] for x in cl if u.filter(pl.col("add1") == x).height >= 500}, "FRANCE copy-count (accepted links)")
