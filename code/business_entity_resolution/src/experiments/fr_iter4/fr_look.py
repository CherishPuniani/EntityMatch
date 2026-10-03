"""Eyeball French S1s with their candidates (final v3 decisions) to find normalisation gaps."""
import polars as pl, sys
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NS1 = int(sys.argv[2]) if len(sys.argv) > 2 else 12
recs = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "entity_id", "business_name", "business_address")
    for s in (1, 2, 3)])
fr = recs.filter(pl.col("id") < 20_000_000_000).select(pl.col("id").alias("s1"))
sc = pl.read_parquet(f"{R}/output_final_v3/scores.parquet").join(fr, on="s1")
m = pl.read_csv(f"{R}/output_final_v3/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
acc = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
       .filter(pl.col("matched_entity_ids") != "").select(
           (pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"),
           (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000)
            + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t"), pl.lit(1).alias("acc")))
sc = sc.join(acc, on=["s1", "t"], how="left").with_columns(pl.col("acc").fill_null(0))
tr = pl.read_parquet(f"{R}/work2/test_scores_tree_unseen.parquet", columns=["s1", "t", "p2"]).rename({"p2": "p2tree"})
sc = sc.join(tr, on=["s1", "t"], how="left")
info = {r[0]: (r[2], r[3]) for r in recs.iter_rows()}
pick = fr.sample(NS1, seed=SEED)["s1"].to_list()
for s1 in pick:
    n, a = info[s1]
    print(f"\n### S1 {n} | {a}")
    c = sc.filter((pl.col("s1") == s1) & (pl.col("p2n") >= 0.01)).sort("p2n", descending=True)
    for t, p2, p2n, ac, pt in c.select("t", "p2", "p2n", "acc", "p2tree").iter_rows():
        tn, ta = info.get(t, ("?", "?"))
        print(f"  {'ACC' if ac else '   '} p2n={p2n:.3f} tree={pt if pt is None else round(pt, 3)}  {tn[:45]:45s} | {ta[:80]}")
