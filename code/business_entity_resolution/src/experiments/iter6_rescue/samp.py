import polars as pl, sys
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
b=pl.read_parquet(SP+'emp_cls_test.parquet').filter((pl.col('acls')==sys.argv[6])&(pl.col('country')==sys.argv[1])&(pl.col('p2n')>float(sys.argv[2]))&(pl.col('p2n')<=float(sys.argv[3])))
u=b.sample(int(sys.argv[4]),seed=int(sys.argv[5]))
ids=set(u['s1'].to_list())|set(u['t'].to_list())
raw=pl.concat([pl.read_csv(f'dataset/test/test_source{s}.tsv',separator='\t',quote_char=None).select((pl.col('entity_id').str.slice(3).cast(pl.Int64)+s*10_000_000_000).alias('id'),'business_name','business_address') for s in (1,2,3)]).filter(pl.col('id').is_in(list(ids)))
info={r[0]:(r[1] or '',r[2] or '') for r in raw.iter_rows()}
for s1,t,p,tb in u.select('s1','t','p2n','tbest').iter_rows():
    a,c=info[s1],info[t]
    print(f'{p:.3f} tb={tb} S1 {a[0][:40]:40s}| {a[1][:60]}\n            T {c[0][:40]:40s}| {c[1][:60]}')
