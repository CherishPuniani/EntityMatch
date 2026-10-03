"""Duplicate consistency: exact-duplicate target records (same name+address+country) always belong to one S1 in train
(60,319/60,319 groups). usage: dupfix.py <in_dir> <out_dir>. For test groups where some members are linked to one S1 and others to none, link the rest to that S1."""
import polars as pl, sys, shutil, os
D = "dataset"   # run from the challenge root
IN, OUT = sys.argv[1], sys.argv[2]
T = pl.concat([pl.read_csv(f"{D}/test/test_source{i}.tsv", separator="\t", quote_char=None, schema_overrides={"entity_id": pl.Utf8}).select(
    pl.col("entity_id").alias("t"), pl.concat_str([pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars(),
                                                  pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars(), "country"], separator="|").alias("k")) for i in (2, 3)])
m = pl.read_csv(f"{IN}/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8}).with_columns(pl.col("matched_entity_ids").fill_null(""))
L = m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").select(
    pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("t"))
g = T.with_columns(pl.len().over("k").alias("gs")).filter(pl.col("gs") >= 2).join(L, on="t", how="left")
grp = g.group_by("k").agg(pl.col("s1").drop_nulls().unique().alias("s1s"), pl.col("s1").null_count().alias("nun"))
fix = grp.filter((pl.col("s1s").list.len() == 1) & (pl.col("nun") > 0)).select("k", pl.col("s1s").list.first().alias("s1"))
add = g.filter(pl.col("s1").is_null()).select("t", "k").join(fix, on="k").select("s1", "t")
print("links added:", add.height)
L2 = pl.concat([L, add])
agg = L2.group_by("s1", maintain_order=True).agg(pl.col("t").str.join(","))
o = m.select("source1_entity_id").join(agg.rename({"s1": "source1_entity_id", "t": "matched_entity_ids"}), on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/matching_results.tsv", "w", encoding="utf-8") as fh:
    fh.write("source1_entity_id\tmatched_entity_ids\n")
    for a, b in o.iter_rows():
        fh.write(f"{a}\t{b}\n")
# candidate file must contain every matched pair: append added targets to the S1's candidate list if missing
c = pl.read_csv(f"{IN}/candidate_pairs.tsv", separator="\t", schema_overrides={"candidate_entity_ids": pl.Utf8}).with_columns(pl.col("candidate_entity_ids").fill_null(""))
ca = add.group_by("s1").agg(pl.col("t").str.join(",").alias("extra")).rename({"s1": "source1_entity_id"})
c = c.join(ca, on="source1_entity_id", how="left").with_columns(
    pl.when(pl.col("extra").is_null()).then(pl.col("candidate_entity_ids"))
    .when(pl.col("candidate_entity_ids") == "").then(pl.col("extra"))
    .otherwise(pl.concat_str(["candidate_entity_ids", "extra"], separator=",")).alias("candidate_entity_ids")).drop("extra")
c = c.with_columns(pl.col("candidate_entity_ids").str.split(",").list.unique(maintain_order=True).list.join(","))
with open(f"{OUT}/candidate_pairs.tsv", "w", encoding="utf-8") as fh:
    fh.write("source1_entity_id\tcandidate_entity_ids\n")
    for a, b in c.iter_rows():
        fh.write(f"{a}\t{b}\n")
