import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
P='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/'
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
def tid(e): return (pl.when(e.str.starts_with('S2')).then(20_000_000_000).when(e.str.starts_with('S3')).then(30_000_000_000).otherwise(10_000_000_000) + e.str.slice(3).cast(pl.Int64))
m=pl.read_csv('output_final8/matching_results.tsv',separator='\t',schema_overrides={'matched_entity_ids':pl.String})
L=(m.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='')
   .select(tid(pl.col('source1_entity_id')).alias('s1'),tid(pl.col('matched_entity_ids')).alias('t')))
L.write_parquet(SP+'links8.parquet')
allS=m.select(tid(pl.col('source1_entity_id')).alias('s1'))
cty=pl.read_parquet('work/p_test_s1.parquet',columns=['id','country']).select(tid(pl.col('id')).alias('s1'),'country')
emp=allS.join(L.select('s1').unique(),on='s1',how='anti').join(cty,on='s1')
sc=pl.read_parquet(P+'final7/output_final7/scores.parquet')
claimed=L.select('t').unique().with_columns(cl=pl.lit(1))
b=(sc.join(emp.select('s1'),on='s1',how='semi').sort('p2n',descending=True).group_by('s1').head(1)
   .join(claimed,on='t',how='left').with_columns(pl.col('cl').fill_null(0)).join(cty,on='s1'))
# also: is this S1 the target's best S1 overall?
tb=sc.group_by('t').agg(tmax=pl.col('p2n').max())
b=b.join(tb,on='t').with_columns(tbest=(pl.col('p2n')>=pl.col('tmax')).cast(pl.Int8))
b.write_parquet(SP+'emp_best_test.parquet')
N=cty.group_by('country').agg(N=pl.len())
bins=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
b=b.with_columns(bin=pl.col('p2n').cut(bins))
r=b.group_by('country','bin','cl').agg(k=pl.len()).join(N,on='country').with_columns(per1k=(pl.col('k')*1000/pl.col('N')).round(3))
print(r.pivot(on='country',index=['bin','cl'],values='per1k').sort('cl','bin'))
print(b.group_by('country').agg(pl.len()).join(N,on='country').with_columns(per1k=pl.col('len')*1000/pl.col('N')))
