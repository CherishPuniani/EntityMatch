"""Unclaimed target records (not linked to any S1) that have the SAME normalised core name as an S1 and the S1's house
number + street core, but are not among that S1's candidates. Train (labels) calibrates: true rate of such pairs.
usage: orphan.py train|test"""
import polars as pl, sys, re
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
split = sys.argv[1]
pref = {"train": f"{R}/work", "test": f"{SP}/fr6/work_unseen"}[split]
P = pl.concat([pl.read_parquet(f"{pref}/p_{split}_s{s}.parquet", columns=["id", "country", "core", "acore"]).select(
    (pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "country",
    pl.col("core").list.unique().list.sort().list.join(" ").alias("ck"),
    pl.col("acore").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first().alias("hn"),
    pl.col("acore").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).alias("st")) for s in (1, 2, 3)])
P = P.filter((pl.col("ck").str.len_chars() >= 3) & pl.col("hn").is_not_null())
S = P.filter(pl.col("id") < 20_000_000_000).rename({"id": "s1", "st": "st1"})
T = P.filter(pl.col("id") >= 20_000_000_000).rename({"id": "t", "st": "st2"})
if split == "train":
    cand = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "y", "p2"])
    claimed = cand.filter(pl.col("p2") >= 0.7).select("t").unique()
    gt = pl.read_parquet(f"{R}/work/gt_pairs_int.parquet").with_columns(pl.lit(1).alias("gt"))
    keep = pl.read_parquet(f"{R}/work/keep_trainD.parquet").rename({"id": "s1"}); S = S.join(keep, on="s1")
else:
    cand = pl.read_parquet(f"{SP}/fr6/output_fr6a/scores.parquet", columns=["s1", "t", "p2n"])
    m = pl.read_csv(f"{SP}/fr6/output_fr6a/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    claimed = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
               .select((pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000) + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t")))
x = S.join(T, on=["country", "ck", "hn"]).join(claimed, on="t", how="anti").join(cand.select("s1", "t"), on=["s1", "t"], how="anti")
x = x.with_columns((pl.col("st1").list.set_intersection("st2").list.len() / pl.max_horizontal(pl.col("st1").list.len(), 1)).alias("stov"))
x = x.filter(pl.col("stov") >= 0.5).with_columns(pl.len().over("t").alias("n_s1_for_t"))
if split == "train":
    x = x.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("gt").fill_null(0))
    print(x.group_by("country", pl.col("n_s1_for_t") == 1).agg(pl.len(), pl.col("gt").mean().round(3).alias("true")).sort("country"))
else:
    ns = S.group_by("country").len()
    print(x.group_by("country", pl.col("n_s1_for_t") == 1).agg(pl.len()).join(ns.rename({"len": "nS1"}), on="country").with_columns((pl.col("len") * 1000 / pl.col("nS1")).round(2).alias("per1k")).sort("country"))
if split == "test" and len(sys.argv) > 2:
    u = x.filter((pl.col("n_s1_for_t") == 1) & (pl.col("country") == "France")).sample(12, seed=int(sys.argv[2]))
    raw = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select(
        (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "business_name", "business_address") for s in (1, 2, 3)])
    info = {r[0]: (r[1], r[2]) for r in raw.iter_rows()}
    sc = cand
    for s1, t in u.select("s1", "t").iter_rows():
        a, b = info[s1], info[t]
        best = sc.filter(pl.col("t") == t).sort("p2n", descending=True).head(1).rows()
        print(f"S1 {a[0][:36]:36s} | {a[1][:62]}\n T {b[0][:36]:36s} | {b[1][:62]}   (t's best cand: {[(info.get(r[0],('?',''))[0][:25], round(r[2],3)) for r in best]})")
