"""Two-stage LightGBM like train_stages.py (same folds, subsample, hyper-parameters, seeds) but WITHOUT the
blocking-strength features (stage-0 ranker score/rank, IDF key sums, pool size, and the target-competition values
derived from them). Motivation: for French same-address copies the ranker score is depressed by generic name tokens
(median rs 0.53 vs 0.92 US), which the US/India-trained tree reads as weak evidence. Outputs to work2/.
env: TAG, SUB, FEAT, ATTRS, EXTRA_PAIRFEATS, NR1, NR2"""
import polars as pl, numpy as np, lightgbm as lgb, sys, time, os, pickle, gc
sys.path.insert(0, "work")
import stages as S
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
NF = 4; SUB = float(os.environ.get("SUB", "1.0")); NR1 = int(os.environ.get("NR1", "800")); NR2 = int(os.environ.get("NR2", "600"))
TAG = os.environ["TAG"]; MD = f"work2/models_{TAG}"; os.makedirs(MD, exist_ok=True)
BLOCK = {"bscore", "nkeys", "brank", "t_best", "t_rank", "t_bbest", "t_brank", "rs", "rrank", "npool", "wmax",
         "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "b_s1rel", "t_gap", "t_bgap", "rs_s1rel"}
f = S.load_feats(os.environ.get("FEAT", "work/feat_train"), "train", os.environ.get("ATTRS"))
gt = pl.read_parquet("work/gt_pairs_int.parquet").with_columns(pl.lit(1).cast(pl.Int8).alias("y"))
f = f.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
f = S.stage1_context(f)
f = f.with_columns((pl.col("s1").hash(seed=11) % NF).cast(pl.Int8).alias("fold"), ((pl.col("s1").hash(seed=23) % 1000) < SUB * 1000).alias("sub"))
F1 = [c for c in S.feat_cols(f, extra_drop=("sub",)) if c not in BLOCK]
log(f"loaded {f.shape}; stage-1 features {len(F1)}")
X = f.select([pl.col(c).cast(pl.Float32) for c in F1]).to_numpy()
y = f["y"].to_numpy(); fold = f["fold"].to_numpy(); sub = f["sub"].to_numpy()
params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, num_threads=30, verbose=-1, seed=1, max_bin=255)
p1 = np.zeros(len(f), np.float32); imp = np.zeros(len(F1))
for k in range(NF):
    tr = (fold != k) & sub
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=F1), NR1)
    p1[fold == k] = m.predict(X[fold == k], num_threads=30); imp += m.feature_importance("gain")
    m.save_model(f"{MD}/s1_f{k}.txt"); log(f"stage1 fold {k}")
del X; gc.collect()
o = np.argsort(-imp); log("stage1 importance: " + str([(F1[i], round(imp[i] / imp.sum(), 4)) for i in o[:25]]))
f = f.with_columns(pl.Series("p1", p1))
f = S.stage2_context(f, "p1")
y = f["y"].to_numpy(); fold = f["fold"].to_numpy(); sub = f["sub"].to_numpy()
F2 = [c for c in S.feat_cols(f, extra_drop=("sub",)) if c not in BLOCK] + ["p1"]
X = f.select([pl.col(c).cast(pl.Float32) for c in F2]).to_numpy()
keep = f.select("s1", "t", "y", "fold", "p1"); del f; gc.collect()
params2 = dict(params, num_leaves=127)
p2 = np.zeros(X.shape[0], np.float32); imp = np.zeros(len(F2))
for k in range(NF):
    tr = (fold != k) & sub
    m = lgb.train(params2, lgb.Dataset(X[tr], y[tr], feature_name=F2), NR2)
    p2[fold == k] = m.predict(X[fold == k], num_threads=30); imp += m.feature_importance("gain")
    m.save_model(f"{MD}/s2_f{k}.txt"); log(f"stage2 fold {k}")
o = np.argsort(-imp); log("stage2 importance: " + str([(F2[i], round(imp[i] / imp.sum(), 4)) for i in o[:25]]))
pickle.dump({"F1": F1, "F2": F2}, open(f"{MD}/cols.pkl", "wb"))
keep.with_columns(pl.Series("p2", p2)).write_parquet(f"work2/oof_{TAG}.parquet"); log("done")
