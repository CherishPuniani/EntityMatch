import polars as pl, sys, time
sys.path.insert(0,"work"); import block
split=sys.argv[1]; cap=int(sys.argv[2]); K=int(sys.argv[3])
keep=sys.argv[4] if len(sys.argv)>4 else None; tag=sys.argv[5] if len(sys.argv)>5 else split
ids=pl.read_parquet(keep)["id"] if keep else None
pairs=block.run(split,cap=cap,K=K,s1_ids=ids,log=lambda m: print(m,flush=True))
pass
pairs.write_parquet(f"work/pairs_{tag}.parquet")
print("done",len(pairs),flush=True)
