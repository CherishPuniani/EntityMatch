"""Final decision step. Countries with training labels (US, India): per-S1 expected-F0.5 decoding (efdec.py) on the
corrected probabilities p2n (OOF-validated: +0.00006 [+0.00003, +0.00009] over threshold 0.7, efdec_oof.py). Countries
without labels keep the input's decisions (p2n >= 0.7, best S1 per target).
usage (from the challenge root, scripts copied to work/): final_ef.py <in_dir with scores.parquet> <out_dir>"""
import polars as pl, sys, os
sys.path.insert(0, "work")
from ids import to_str
import efdec
IN, OUT = sys.argv[1], sys.argv[2]
seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
sc = pl.read_parquet(f"{IN}/scores.parquet")
lab = s1te.filter(pl.col("country").is_in(list(seen))).select("s1")
sel_lab = efdec.decode(sc.join(lab, on="s1").select("s1", "t", pl.col("p2n").alias("p")))
old = sc.with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt")).filter(
    (pl.col("p2n") >= 0.7) & (pl.col("rt") == 1)).select("s1", "t")
new = pl.concat([old.join(lab, on="s1", how="anti"), sel_lab]).join(sc.select("s1", "t", "p2n"), on=["s1", "t"])
assert new.select("t").is_duplicated().sum() == 0
print("vs input: added", new.join(old, on=["s1", "t"], how="anti").join(s1te.select("s1", "country"), on="s1").group_by("country").len().rows(),
      "removed", old.join(new, on=["s1", "t"], how="anti").join(s1te.select("s1", "country"), on="s1").group_by("country").len().rows())
os.makedirs(OUT, exist_ok=True)
S1 = s1te.select(pl.col("entity_id").alias("source1_entity_id"))


def write(pairs, pcol, colname, path):
    agg = (pairs.sort(["s1", pcol], descending=[False, True]).with_columns(to_str("s1"), to_str("t"))
           .group_by("s1", maintain_order=True).agg(pl.col("t").str.join(",").alias(colname)).rename({"s1": "source1_entity_id"}))
    o = S1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col(colname).fill_null(""))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{colname}\n")
        for a, b in o.iter_rows():
            fh.write(f"{a}\t{b}\n")


write(new, "p2n", "matched_entity_ids", f"{OUT}/matching_results.tsv")
write(sc, "p2n", "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
sc.write_parquet(f"{OUT}/scores.parquet")
