"""Test inference for the CE-augmented stage 2: baseline stage-1 p1 (from test_scores_<BASE_TAG>), CE scores on the
gated pairs, CE context, stage-2 models of TAG, threshold + target exclusivity, then both submission TSVs.
env: TAG, BASE_TAG, CE_SCORES (test; s1,t,ce), TH, OUT
"""
import polars as pl, numpy as np, lightgbm as lgb, sys, os, pickle, time
sys.path.insert(0, "work")
import stages as S
from ids import to_str
from train_stage2_ce import ce_context

TAG = os.environ["TAG"]; BASE = os.environ.get("BASE_TAG", "base"); TH = float(os.environ.get("TH", "0.7"))
OUT = os.environ.get("OUT", "output")
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
cols = pickle.load(open(f"work/models_{TAG}/cols.pkl", "rb"))
f = S.stage1_context(S.load_feats(os.environ.get("FEAT", "work/feat_test"), "test"))
bs = pl.read_parquet(f"work/test_scores_{BASE}.parquet", columns=["s1", "t", "p1"])
n0 = len(f)
f = f.join(bs, on=["s1", "t"], how="inner", maintain_order="left"); assert len(f) == n0 == len(bs)
if "ce" in cols["F2"]:
    f = ce_context(f.join(pl.read_parquet(os.environ["CE_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left",
                          maintain_order="left"))
    log(f"CE coverage {f['ce'].is_not_null().mean():.4f}")
f = S.stage2_context(f, "p1")
res = f.select("s1", "t", "p1")
f = f.select([pl.col(c).cast(pl.Float32) for c in cols["F2"]])     # narrow float32 frame (no float64 copy)
import gc; gc.collect()
models = [lgb.Booster(model_file=f"work/models_{TAG}/s2_f{k}.txt") for k in range(4)]
p2 = np.zeros(len(f), np.float32); CH = 5_000_000
for i in range(0, len(f), CH):                                   # row chunks bound the matrix memory
    X = f.slice(i, CH).to_numpy()
    p2[i:i + CH] = np.mean([m.predict(X, num_threads=30) for m in models], axis=0)
del f, X
res = res.with_columns(pl.Series("p2", p2))
res.write_parquet(f"work/test_scores_{TAG}.parquet")
log("scored")
sel = res.with_columns(pl.col("p2").rank("ordinal", descending=True).over("t").alias("rt")).filter(
    (pl.col("p2") >= TH) & (pl.col("rt") == 1))
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
