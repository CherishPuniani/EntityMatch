import polars as pl, sys
sys.path.insert(0,'/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad')
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
R='/home2/home/amritanshu_t/amazon-mlc-26/run'
OP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/'
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
def tid(e): return (pl.when(e.str.starts_with('S2')).then(20_000_000_000).when(e.str.starts_with('S3')).then(30_000_000_000).otherwise(10_000_000_000) + e.str.slice(3).cast(pl.Int64))
m=pl.read_csv(SP+'final9/output_final9/matching_results.tsv',separator='\t',schema_overrides={'matched_entity_ids':pl.String})
L=(m.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='')
   .select(tid(pl.col('source1_entity_id')).alias('s1'),tid(pl.col('matched_entity_ids')).alias('t')))
L.write_parquet(SP+'links9.parquet')
nl=L.group_by('s1').agg(nl=pl.len())
sc=pl.read_parquet(OP+'final7/output_final7/scores.parquet',columns=['s1','t','p2n'])
sc=sc.with_columns(tmax=pl.col('p2n').max().over('t'))
pool=sc.filter((pl.col('p2n')>=0.02)&(pl.col('p2n')>=pl.col('tmax'))).join(L.select('t').unique(),on='t',how='anti')
exec(open(SP+'cls_lib.py').read())
P=pl.concat([prep(OP+'fr6/work_unseen','test').filter(pl.col('country')=='France'),prep(R+'/work','test').filter(pl.col('country')!='France')])
cty=P.filter(pl.col('id')<20_000_000_000).select(pl.col('id').alias('s1'),'country')
x=classify(pool,P).join(cty,on='s1').join(nl,on='s1',how='left').with_columns(pl.col('nl').fill_null(0))
x=x.with_columns(bin=pl.col('p2n').cut([0.05,0.1,0.2,0.3,0.4,0.5,0.7,0.9]),emp=(pl.col('nl')==0))
x.write_parquet(SP+'pool9.parquet')
N=cty.group_by('country').agg(N=pl.len())
r=x.group_by('country','acls','bin').agg(k=pl.len()).join(N,on='country').with_columns(per1k=(pl.col('k')*1000/pl.col('N')).round(2))
print(r.pivot(on='country',index=['acls','bin'],values='per1k').sort('acls','bin'))
