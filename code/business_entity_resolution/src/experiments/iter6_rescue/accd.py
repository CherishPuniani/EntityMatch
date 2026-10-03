import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
R='/home2/home/amritanshu_t/amazon-mlc-26/run'
OP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/'
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
exec(open(SP+'cls_lib.py').read())
L=pl.read_parquet(SP+'links9.parquet')
sc=pl.read_parquet(OP+'final7/output_final7/scores.parquet',columns=['s1','t','p2n']).join(L,on=['s1','t'],how='semi')
P=pl.concat([prep(OP+'fr6/work_unseen','test').filter(pl.col('country')=='France'),prep(R+'/work','test').filter(pl.col('country')!='France')])
cty=P.filter(pl.col('id')<20_000_000_000).select(pl.col('id').alias('s1'),'country')
nl=L.group_by('s1').agg(nl=pl.len())
x=classify(sc,P).join(cty,on='s1').join(nl,on='s1').with_columns(bin=pl.col('p2n').cut([0.7,0.9,0.99,0.999]))
x.write_parquet(SP+'acc9.parquet')
N=cty.group_by('country').agg(N=pl.len())
r=x.group_by('country','acls','bin').agg(k=pl.len()).join(N,on='country').with_columns(per1k=(pl.col('k')*1000/pl.col('N')).round(2))
print(r.pivot(on='country',index=['acls','bin'],values='per1k').sort('acls','bin'))
print('copy-count (other links = nl-1) of accepted links by class, France:')
print(x.filter(pl.col('country')=='France').group_by('acls').agg(n=pl.len(),cc=(pl.col('nl')-1).mean().round(3)).sort('acls'))
