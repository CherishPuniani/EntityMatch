import polars as pl, numpy as np, lightgbm as lgb, sys
p=pl.read_parquet(sys.argv[1]); out=sys.argv[2]
gt=pl.read_parquet("work/gt_pairs_int.parquet").with_columns(pl.lit(1).cast(pl.Int8).alias("y"))
p=p.join(gt,on=["s1","t"],how="left",maintain_order="left").with_columns(pl.col("y").fill_null(0))
p=p.with_columns((pl.col("bscore")/pl.col("bscore").max().over("s1")).alias("brel"),pl.col("noaddr").cast(pl.Int8))
F=["bscore","nkeys","wmax"]+[f"w{k}" for k in range(8)]+["noaddr","npool","brank","brel"]
prm=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=500,num_threads=30,verbose=-1,seed=3)
m=lgb.train(prm,lgb.Dataset(p.select(F).to_numpy().astype(np.float32),p["y"].to_numpy(),feature_name=F),200)
m.save_model(out); print("saved")
