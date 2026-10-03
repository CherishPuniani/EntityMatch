"""Stage 2 of the two-stage LightGBM with added cross-encoder (CE) features, re-using the baseline's OOF p1, folds,
subsample and hyperparameters, so the only change vs the baseline stage 2 is the CE feature block.

env: BASE_TAG (models/oof of the baseline, default base), TAG (output), CE_SCORES (s1,t,ce; OOF on train),
     NOCE=1 (baseline stage 2 only, e.g. with another SEED to measure the training-noise floor), SEED, SUB, NR2,
     FEAT, ATTRS as in train_stages.py.
"""
import polars as pl, numpy as np, lightgbm as lgb, sys, time, os, pickle
sys.path.insert(0, "work")
import stages as S

t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
NF = 4
SUB = float(os.environ.get("SUB", "0.5")); NR2 = int(os.environ.get("NR2", "600")); SEED = int(os.environ.get("SEED", "1"))
TAG = os.environ["TAG"]; BASE = os.environ.get("BASE_TAG", "base"); NOCE = os.environ.get("NOCE") == "1"
os.makedirs(f"work/models_{TAG}", exist_ok=True)
CE_COLS = ["ce", "ce_rank_s1", "ce_max_s1", "ce_gap_s1", "ce_npos_s1", "ce_t_other", "ce_t_margin"]
# CE_WITHIN=1: only within-S1 CE features. All pairs of one S1 lie in one half and are scored by the CE trained on the
# other half, so these are strictly out-of-fold; the cross-S1 ones (ce_t_*) mix rivals scored by the other CE.
CE_USE = CE_COLS[:5] if os.environ.get("CE_WITHIN") == "1" else CE_COLS


def ce_context(f):
    """CE score plus its within-S1 and target-competition context (defined on scored pairs; null elsewhere)."""
    sc = f.filter(pl.col("ce").is_not_null())
    top = sc.group_by("t").agg(pl.col("ce").top_k(2).alias("_tk")).select(
        "t", pl.col("_tk").list.get(0).alias("_m1"), pl.col("_tk").list.get(1, null_on_oob=True).alias("_m2"))
    f = f.with_columns(
        pl.col("ce").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("ce_rank_s1"),
        pl.col("ce").max().over("s1").alias("ce_max_s1"),
        (pl.col("ce") > 0).cast(pl.Float32).sum().over("s1").alias("ce_npos_s1"),
        pl.col("ce").rank("ordinal", descending=True).over("t").alias("_rt"))
    f = f.join(top, on="t", how="left", maintain_order="left").with_columns(
        pl.when(pl.col("ce").is_null()).then(None).when(pl.col("_rt") == 1).then(pl.col("_m2")).otherwise(pl.col("_m1")).alias("ce_t_other"))
    f = f.with_columns((pl.col("ce") - pl.col("ce_max_s1")).alias("ce_gap_s1"),
                       (pl.col("ce") - pl.col("ce_t_other")).alias("ce_t_margin")).drop("_m1", "_m2", "_rt")
    return f.with_columns([pl.col(c).cast(pl.Float32) for c in CE_COLS])


if __name__ == "__main__":
    f = S.load_feats(os.environ.get("FEAT", "work/feat_trainD"), "train", os.environ.get("ATTRS", "trainD"))
    f = S.stage1_context(f)
    oof = pl.read_parquet(f"work/oof_{BASE}.parquet", columns=["s1", "t", "y", "fold", "p1"])
    n0 = len(f)
    f = f.join(oof, on=["s1", "t"], how="inner", maintain_order="left")
    assert len(f) == n0 == len(oof), (n0, len(f), len(oof))
    f = f.with_columns(((pl.col("s1").hash(seed=23) % 1000) < SUB * 1000).alias("sub"))
    if not NOCE:
        ce = pl.read_parquet(os.environ["CE_SCORES"], columns=["s1", "t", "ce"])
        f = ce_context(f.join(ce, on=["s1", "t"], how="left", maintain_order="left"))
        log(f"CE coverage {f['ce'].is_not_null().mean():.4f} of pairs, {f.filter(pl.col('y')==1)['ce'].is_not_null().mean():.4f} of positives")
    f = S.stage2_context(f, "p1")
    y = f["y"].to_numpy(); fold = f["fold"].to_numpy(); sub = f["sub"].to_numpy()
    base_cols = pickle.load(open(f"work/models_{BASE}/cols.pkl", "rb"))
    F2 = list(base_cols["F2"]) + ([] if NOCE else CE_USE)
    log(f"loaded {f.shape}; {len(F2)} stage-2 features")
    # cast inside polars so to_numpy yields float32 directly (a mixed-dtype select would materialise float64 first),
    # then drop the wide frame before training: keeps the 68.6M-row union well under the 160 GB limit
    X = f.select([pl.col(c).cast(pl.Float32) for c in F2]).to_numpy()
    keep = f.select("s1", "t", "y", "fold", "p1"); del f
    import gc; gc.collect()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, num_threads=30, verbose=-1, seed=SEED, max_bin=255)
    p2 = np.zeros(X.shape[0], np.float32); imp = np.zeros(len(F2))
    for k in range(NF):
        tr = (fold != k) & sub
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=F2), NR2)
        p2[fold == k] = m.predict(X[fold == k], num_threads=30)
        imp += m.feature_importance("gain")
        m.save_model(f"work/models_{TAG}/s2_f{k}.txt")
        log(f"stage2 fold {k}")
    o = np.argsort(-imp)
    log("stage2 importance: " + str([(F2[i], round(imp[i] / imp.sum(), 4)) for i in o[:25]]))
    pickle.dump({"F1": base_cols["F1"], "F2": F2}, open(f"work/models_{TAG}/cols.pkl", "wb"))
    keep.with_columns(pl.Series("p2", p2)).write_parquet(f"work/oof_{TAG}.parquet")
    log("done")
