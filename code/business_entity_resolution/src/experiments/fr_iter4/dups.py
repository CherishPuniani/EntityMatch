"""Exact-duplicate target records (same lowercased name+address+country, S2/S3). Train: do duplicates share their S1?
Test: how consistent is our assignment within duplicate groups?"""
import polars as pl, sys
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"; D = f"{R}/dataset"
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
def tg(split):
    return pl.concat([pl.read_csv(f"{D}/{split}/{split}_source{i}.tsv", separator="\t", quote_char=None, schema_overrides={"entity_id": pl.Utf8}).select(
        pl.col("entity_id").alias("t"), "country", pl.concat_str([pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars(),
                                                              pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars(), "country"], separator="|").alias("k")) for i in (2, 3)])
def links(path, c1="source1_entity_id", c2="matched_entity_ids"):
    return pl.read_csv(path, separator="\t", schema_overrides={c2: pl.Utf8}).with_columns(pl.col(c2).fill_null("").str.split(",")).explode(c2).filter(pl.col(c2) != "").select(pl.col(c1).alias("s1"), pl.col(c2).alias("t"))
for split, lpath in (("train", f"{D}/train/train_ground_truth.tsv"), ("test", f"{SP}/final7/output_final7/matching_results.tsv")):
    T = tg(split)
    g = T.with_columns(pl.len().over("k").alias("gs")).filter(pl.col("gs") >= 2)
    L = links(lpath)
    x = g.join(L, on="t", how="left")
    s = x.group_by("k").agg(pl.len().alias("n"), pl.col("s1").drop_nulls().n_unique().alias("n_s1"), pl.col("s1").null_count().alias("n_unlinked"), pl.col("country").first())
    print(f"{split}: dup groups {s.height} ({g.height} records); groups all linked to one S1: {(s['n_s1'].eq(1) & s['n_unlinked'].eq(0)).sum()}, "
          f"none linked: {(s['n_s1'].eq(0)).sum()}, partly linked (1 S1 + unlinked): {(s['n_s1'].eq(1) & s['n_unlinked'].gt(0)).sum()}, multi-S1: {(s['n_s1'] > 1).sum()}")
    print("   by country partly-linked:", s.filter(pl.col("n_s1").eq(1) & pl.col("n_unlinked").gt(0)).group_by("country").agg(pl.len(), pl.col("n_unlinked").sum()).rows())
