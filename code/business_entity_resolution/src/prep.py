import polars as pl, sys, time, os
from multiprocessing import Pool
sys.path.insert(0,"work")
from norm import name_forms, addr_forms, LEGAL, HONOR, NOISE, ADDR_DROP, STATE_CODES

def proc(rows):
    out=[]
    for eid,n,a,c in rows:
        nf,nt,alias,oov=name_forms(n)
        core=[t for t in nt if t not in LEGAL and t not in HONOR and t not in NOISE]
        af,comps,at=addr_forms(a)
        acore=[t for t in at if t not in ADDR_DROP and t not in STATE_CODES]
        out.append((eid,c,nf,nt,core,alias[0] if alias else None,alias[1] if alias else None,oov,af,at,acore,len(comps)))
    return out
L=pl.List(pl.Utf8)
cols={"id":pl.Utf8,"country":pl.Utf8,"nf":pl.Utf8,"nt":L,"core":L,"al_pre":pl.Utf8,"al_post":pl.Utf8,"oov":pl.Int32,"af":pl.Utf8,"at":L,"acore":L,"ncomp":pl.Int32}
if __name__=="__main__":
    for split in ["train","test"]:
        for s in [1,2,3]:
            t=time.time()
            df=pl.read_parquet(f"work/{split}_s{s}.parquet")
            rows=df.rows()
            ch=[rows[i:i+20000] for i in range(0,len(rows),20000)]
            with Pool(int(os.environ.get("PREP_PROCS","2"))) as p: res=p.map(proc,ch,chunksize=1)
            flat=[r for part in res for r in part]
            out=pl.DataFrame(flat,schema=cols,orient="row")
            out.write_parquet(f"work/p_{split}_s{s}.parquet")
            print(split,s,len(out),time.time()-t,flush=True)
