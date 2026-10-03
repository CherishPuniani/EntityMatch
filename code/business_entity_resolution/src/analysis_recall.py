"""Candidate-recall diagnostics for a blocking output: recall@K, complete-set recall, raw-pool ceiling on the ranker
sample, and a categorisation of missed true pairs.
usage: analysis_recall.py <pairs.parquet> <keep.parquet> [rawpool.parquet]"""
import sys, polars as pl
from rapidfuzz import process, fuzz

pairs = pl.read_parquet(sys.argv[1], columns=["s1", "t", "rrank"])
keep = pl.read_parquet(sys.argv[2]).rename({"id": "s1"})
gt = pl.read_parquet("work/gt_pairs_int.parquet").join(keep, on="s1")
j = gt.join(pairs, on=["s1", "t"], how="left")
G = len(gt)
print(f"true pairs {G}")
for K in (1, 5, 10, 20, 30):
    print(f"recall@{K}: {(j['rrank'] <= K).sum() / G:.4f}")
cs = j.group_by("s1").agg((pl.col("rrank").is_not_null()).all().alias("all"))
print(f"complete-set recall (non-singleton S1 with every true target retrieved): {cs['all'].mean():.4f}")
if len(sys.argv) > 3:
    rp = pl.read_parquet(sys.argv[3], columns=["s1", "t"])
    smp = rp.select("s1").unique().join(keep, on="s1")
    g2 = gt.join(smp, on="s1")
    print(f"raw-pool ceiling on the ranker sample: {len(g2.join(rp, on=['s1','t'])) / len(g2):.4f} ({len(g2)} true pairs; ranker trained on these S1s)")
miss = j.filter(pl.col("rrank").is_null()).select("s1", "t")
P = {s: pl.read_parquet(f"work/p_train_s{s}.parquet", columns=["id", "country", "nt", "acore"]).with_columns(
    (pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id")) for s in (1, 2, 3)}
R = {s: pl.read_parquet(f"work/train_s{s}.parquet").with_columns(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id")) for s in (1, 2, 3)}
T = pl.concat([P[2], P[3]]).join(pl.concat([R[2], R[3]]).select("id", "business_name", "business_address"), on="id")
S = P[1].join(R[1].select("id", "business_name", "business_address"), on="id")
m = miss.join(S.rename({c: c + "_1" for c in S.columns if c != "id"}), left_on="s1", right_on="id").join(
    T.rename({c: c + "_2" for c in T.columns if c != "id"}), left_on="t", right_on="id")
ns1 = m["nt_1"].list.join(" ").to_list(); ns2 = m["nt_2"].list.join(" ").to_list()
as1 = m["acore_1"].list.join(" ").to_list(); as2 = m["acore_2"].list.join(" ").to_list()
m = m.with_columns(pl.Series("nsim", process.cpdist(ns1, ns2, scorer=fuzz.token_set_ratio, workers=4)),
                   pl.Series("asim", process.cpdist(as1, as2, scorer=fuzz.token_set_ratio, workers=4)))
m = m.with_columns(pl.when(pl.col("business_address_2").str.strip_chars() == "").then(pl.lit("empty-address target"))
                   .when(pl.col("business_name_2").str.contains(r"[ऀ-෿]")).then(pl.lit("native-script name"))
                   .when(pl.col("business_name_2").str.contains(r"(?i)\.com|www\.|^@")).then(pl.lit("domain-form name"))
                   .when(pl.col("nsim") < 50).then(pl.lit("renamed (name sim<50)"))
                   .when(pl.col("asim") < 50).then(pl.lit("similar name, address sim<50"))
                   .otherwise(pl.lit("similar name + address")).alias("cat"))
print(f"missed true pairs: {len(m)} ({len(m)/G:.4f})")
print(m.group_by("cat").len().sort("len", descending=True).with_columns((pl.col("len") / len(m)).round(3).alias("share")))
print(m.group_by("country_1").len())
m.select("s1", "t", "cat", "nsim", "asim", "business_name_1", "business_address_1", "business_name_2",
         "business_address_2").write_parquet("work/miss_trainD.parquet")
with pl.Config(fmt_str_lengths=60, tbl_width_chars=250, tbl_rows=24):
    print(m.filter(pl.col("cat") != "empty-address target").sample(12, seed=3).select(
        "cat", "business_name_1", "business_address_1", "business_name_2", "business_address_2"))
