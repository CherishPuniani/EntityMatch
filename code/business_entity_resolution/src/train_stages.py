"""Two-stage grouped CV on all train pairs. Saves OOF p1/p2 and models."""
import polars as pl, numpy as np, lightgbm as lgb, sys, time, os, pickle
sys.path.insert(0, "work")
import stages as S

t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
NF = 4
SUB = float(os.environ.get("SUB", "0.35"))    # fraction of training-fold S1s used per stage-1 model
NR1 = int(os.environ.get("NR1", "800")); NR2 = int(os.environ.get("NR2", "600"))
TAG = os.environ.get("TAG", "v1")
os.makedirs(f"work/models_{TAG}", exist_ok=True)

f = S.load_feats(os.environ.get("FEAT", "work/feat_train"), "train", os.environ.get("ATTRS"))
gt = pl.read_parquet("work/gt_pairs_int.parquet").with_columns(pl.lit(1).cast(pl.Int8).alias("y"))
f = f.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
f = S.stage1_context(f)
f = f.with_columns((pl.col("s1").hash(seed=11) % NF).cast(pl.Int8).alias("fold"),
                   ((pl.col("s1").hash(seed=23) % 1000) < SUB * 1000).alias("sub"))
log(f"loaded {f.shape}")
F1 = S.feat_cols(f, extra_drop=("sub",))
X = f.select(F1).to_numpy().astype(np.float32)
y = f["y"].to_numpy(); fold = f["fold"].to_numpy(); sub = f["sub"].to_numpy()
params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, num_threads=30, verbose=-1, seed=1, max_bin=255)
p1 = np.zeros(len(f), np.float32)
imp = np.zeros(len(F1))
for k in range(NF):
    tr = (fold != k) & sub
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=F1), NR1)
    p1[fold == k] = m.predict(X[fold == k], num_threads=30)
    imp += m.feature_importance("gain")
    m.save_model(f"work/models_{TAG}/s1_f{k}.txt")
    log(f"stage1 fold {k}")
del X
o = np.argsort(-imp)
log("stage1 importance: " + str([(F1[i], round(imp[i] / imp.sum(), 4)) for i in o[:30]]))
f = f.with_columns(pl.Series("p1", p1))
f = S.stage2_context(f, "p1")
y = f["y"].to_numpy(); fold = f["fold"].to_numpy(); sub = f["sub"].to_numpy()  # re-read after joins
F2 = S.feat_cols(f, extra_drop=("sub",)) + ["p1"]
X = f.select(F2).to_numpy().astype(np.float32)
p2 = np.zeros(len(f), np.float32)
imp = np.zeros(len(F2))
params2 = dict(params, learning_rate=0.05, num_leaves=127)
for k in range(NF):
    tr = (fold != k) & sub
    m = lgb.train(params2, lgb.Dataset(X[tr], y[tr], feature_name=F2), NR2)
    p2[fold == k] = m.predict(X[fold == k], num_threads=30)
    imp += m.feature_importance("gain")
    m.save_model(f"work/models_{TAG}/s2_f{k}.txt")
    log(f"stage2 fold {k}")
o = np.argsort(-imp)
log("stage2 importance: " + str([(F2[i], round(imp[i] / imp.sum(), 4)) for i in o[:30]]))
pickle.dump({"F1": F1, "F2": F2}, open(f"work/models_{TAG}/cols.pkl", "wb"))
f.select("s1", "t", "y", "fold", "p1", pl.Series("p2", p2)).write_parquet(f"work/oof_{TAG}.parquet")
log("done")
