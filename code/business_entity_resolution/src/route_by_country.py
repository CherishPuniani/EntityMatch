"""Combine two scored test sets by S1 country: S1s whose country appears in the training labels use the main
(neural-augmented) p2; S1s of countries absent from training (France in this test set) use the fallback p2 from the
tree-only model. The set of training countries is read from the data (open-set; nothing is hard-coded), and every S1
still gets a prediction. Blocking never crosses countries, so target exclusivity is unaffected by the routing.

usage: route_by_country.py <main_scores.parquet> <fallback_scores.parquet> <out_dir> [th=0.7]
Both score files hold s1, t, p2. Each S1 keeps the candidate set of the model that scores it, so candidate_pairs.tsv
is exactly what the deployed pipeline scored for that S1."""
import sys, os, polars as pl
sys.path.insert(0, "work")
from ids import to_str

main, fb, out = sys.argv[1], sys.argv[2], sys.argv[3]
TH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
s1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
unseen = s1.filter(~pl.col("country").is_in(list(seen)))
A = pl.read_parquet(main, columns=["s1", "t", "p2"]); B = pl.read_parquet(fb, columns=["s1", "t", "p2"])
res = pl.concat([A.join(unseen.select("s1"), on="s1", how="anti"), B.join(unseen.select("s1"), on="s1")])
print(f"training countries {sorted(seen)}; unseen-country S1s routed to fallback: {len(unseen)} "
      f"({unseen['country'].unique().to_list()}); pairs {len(res)}")
sel = res.with_columns(pl.col("p2").rank("ordinal", descending=True).over("t").alias("rt")).filter(
    (pl.col("p2") >= TH) & (pl.col("rt") == 1))
os.makedirs(out, exist_ok=True)
S1 = s1.select(pl.col("entity_id").alias("source1_entity_id"))


def write(pairs, colname, path):
    agg = (pairs.sort(["s1", "p2"], descending=[False, True]).with_columns(to_str("s1"), to_str("t"))
           .group_by("s1", maintain_order=True).agg(pl.col("t").str.join(",").alias(colname))
           .rename({"s1": "source1_entity_id"}))
    o = S1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col(colname).fill_null(""))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{colname}\n")
        for a, b in o.iter_rows():
            fh.write(f"{a}\t{b}\n")
    return o


m = write(sel, "matched_entity_ids", f"{out}/matching_results.tsv")
write(res, "candidate_entity_ids", f"{out}/candidate_pairs.tsv")
res.write_parquet(f"{out}/routed_scores.parquet")
print(f"wrote {len(m)} rows; non-empty {(m['matched_entity_ids'] != '').sum()}; matches {len(sel)}")
