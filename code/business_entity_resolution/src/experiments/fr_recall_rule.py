"""France recall probe: on top of a corrected submission, accept French same-address candidates that keep the S1's
first name token, carry no legal-form change, are not better claimed by another S1, and have p2n >= FLOOR.
usage: fr_recall_rule.py <in_dir with scores.parquet> <out_dir> [floor=0.2]"""
import polars as pl, glob, sys, os
sys.path.insert(0, "work"); sys.path.insert(0, "work")
from lfutil import lf
from ids import to_str
IN, OUT = sys.argv[1], sys.argv[2]; FLOOR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2; TH = 0.7
s1c = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
fr = s1c.filter(pl.col("country") == "France").select("s1")
sc = pl.read_parquet(f"{IN}/scores.parquet")
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_eq", "aempty_2", "acs_tset", "first_eq"]) for d in ["work/feat_test", "work/feat_test_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))]).join(fr, on="s1")
R = pl.concat([pl.read_parquet(f"work/test_s{s}.parquet", columns=["entity_id", "business_name", "country"]).filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), lf("business_name").alias("lf")) for s in (1, 2, 3)])
x = sc.join(fr, on="s1").with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt")).join(f, on=["s1", "t"])
x = x.join(R.rename({"id": "s1", "lf": "l1"}), on="s1").join(R.rename({"id": "t", "lf": "l2"}), on="t")
x = x.with_columns(((pl.col("l1") != "") & (pl.col("l2") != "") & (pl.col("l1") != pl.col("l2"))).alias("chg"))
rescue = x.filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1) & (pl.col("acs_tset") >= 90) & (pl.col("first_eq") == 1) & ~pl.col("chg")
                  & (pl.col("rt") == 1) & (pl.col("p2n") >= FLOOR) & (pl.col("p2n") < TH)).select("s1", "t", pl.lit(True).alias("r"))
res = sc.join(rescue, on=["s1", "t"], how="left").with_columns(pl.when(pl.col("r").fill_null(False)).then(pl.max_horizontal(pl.col("p2n"), pl.lit(0.7001))).otherwise(pl.col("p2n")).alias("p2n")).drop("r")
def select(df): return df.with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt")).filter((pl.col("p2n") >= TH) & (pl.col("rt") == 1))
old, new = select(sc), select(res)
k = s1c.select("s1", "country").join(new.group_by("s1").len(), on="s1", how="left").with_columns(pl.col("len").fill_null(0))
print(f"rescued pairs {len(rescue)} ({len(rescue) * 1000 / len(fr):.1f} per 1k French S1); links {len(old)} -> {len(new)}")
print("matches per S1 / empty:", k.group_by("country").agg(pl.col("len").mean().alias("k"), (pl.col("len") == 0).mean().alias("empty")).sort("country").rows())
os.makedirs(OUT, exist_ok=True)
S1 = s1c.select(pl.col("entity_id").alias("source1_entity_id"))
def write(pairs, colname, path):
    agg = (pairs.sort(["s1", "p2n"], descending=[False, True]).with_columns(to_str("s1"), to_str("t")).group_by("s1", maintain_order=True)
           .agg(pl.col("t").str.join(",").alias(colname)).rename({"s1": "source1_entity_id"}))
    o = S1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col(colname).fill_null(""))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{colname}\n")
        for a, b in o.iter_rows(): fh.write(f"{a}\t{b}\n")
write(new, "matched_entity_ids", f"{OUT}/matching_results.tsv"); write(res, "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
res.write_parquet(f"{OUT}/scores.parquet")
