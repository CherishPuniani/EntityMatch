"""Score test candidate pairs with the two-stage model and write submission files."""
import polars as pl, numpy as np, lightgbm as lgb, sys, os, pickle, time
sys.path.insert(0, "work")
import stages as S
from ids import to_str

TAG = os.environ.get("TAG", "v1"); TH = float(os.environ.get("TH", "0.7")); EXCL = os.environ.get("EXCL", "1") == "1"
OUT = os.environ.get("OUT", "output")
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
cols = pickle.load(open(f"work/models_{TAG}/cols.pkl", "rb"))
f = S.load_feats(os.environ.get("FEAT", "work/feat_test"), "test")
f = S.stage1_context(f)
log(f"loaded {f.shape}")
X = f.select(cols["F1"]).to_numpy().astype(np.float32)
p1 = np.mean([lgb.Booster(model_file=f"work/models_{TAG}/s1_f{k}.txt").predict(X, num_threads=30) for k in range(4)], axis=0)
del X
f = f.with_columns(pl.Series("p1", p1.astype(np.float32)))
f = S.stage2_context(f, "p1")
X = f.select(cols["F2"]).to_numpy().astype(np.float32)
p2 = np.mean([lgb.Booster(model_file=f"work/models_{TAG}/s2_f{k}.txt").predict(X, num_threads=30) for k in range(4)], axis=0)
del X
res = f.select("s1", "t", "p1", pl.Series("p2", p2.astype(np.float32)))
res.write_parquet(f"work/test_scores_{TAG}.parquet")
log("scored")

sel = res.with_columns(pl.col("p2").rank("ordinal", descending=True).over("t").alias("rt"))
sel = sel.filter(pl.col("p2") >= TH)
if EXCL:
    sel = sel.filter(pl.col("rt") == 1)
s1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "source1_entity_id"})
os.makedirs(OUT, exist_ok=True)


def write(pairs, colname, path):
    agg = (pairs.sort(["s1", "p2"], descending=[False, True]).with_columns(to_str("s1"), to_str("t"))
           .group_by("s1", maintain_order=True).agg(pl.col("t").str.join(",").alias(colname))
           .rename({"s1": "source1_entity_id"}))
    out = s1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col(colname).fill_null(""))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{colname}\n")
        for a, b in out.iter_rows():
            fh.write(f"{a}\t{b}\n")
    return out


m = write(sel, "matched_entity_ids", f"{OUT}/matching_results.tsv")
c = write(res, "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
log(f"wrote {len(m)} rows; predicted non-empty {(m['matched_entity_ids'] != '').sum()}; matches {len(sel)}")
