"""Raw candidate pool (all pairs sharing >=1 key) with per-key-type weight sums, for a sample of train S1."""
import polars as pl, numpy as np, sys, time
sys.path.insert(0,"work"); import block
split=sys.argv[1]; frac=float(sys.argv[2]); out=sys.argv[3]; cap=int(sys.argv[4]) if len(sys.argv)>4 else 300
t=time.time()
tgt=pl.concat([block.base(split,2),block.base(split,3)])
tdf=block.df_tables(tgt); N=len(tgt)
noaddr=tgt.select(pl.col("id").alias("id_t"),"noaddr")
kt=pl.concat([block.keys(tgt.slice(i,1_000_000),tdf,"t") for i in range(0,N,1_000_000)]); del tgt
kdf=kt.group_by("h").len("df").filter(pl.col("df")<=cap)
kdf=kdf.with_columns((np.log(N)-pl.col("df").cast(pl.Float64).log()).cast(pl.Float32).alias("w"))
kt=kt.join(kdf.select("h","w"),on="h"); print("kt",len(kt),time.time()-t,flush=True)
s1=block.base(split,1).sample(fraction=frac,seed=99)
ks=block.keys(s1,tdf,"s1")
j=ks.join(kt.drop("ty"),on="h",suffix="_t")
agg=[pl.col("w").sum().alias("bscore"),pl.len().cast(pl.Int16).alias("nkeys"),pl.col("w").max().alias("wmax")]
agg+=[pl.col("w").filter(pl.col("ty")==k).sum().alias(f"w{k}") for k in range(block.NTY)]
pairs=j.group_by("id","id_t").agg(agg).join(noaddr,on="id_t",how="left")
pairs=pairs.with_columns(pl.col("bscore").rank("ordinal",descending=True).over("id").cast(pl.Int32).alias("brank"),
                         pl.len().over("id").cast(pl.Int32).alias("npool"))
pairs=pairs.rename({"id":"s1","id_t":"t"})
pairs.write_parquet(out); print("pairs",len(pairs),"S1",len(s1),time.time()-t,flush=True)
