import polars as pl, sys
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
R='/home2/home/amritanshu_t/amazon-mlc-26/run'
OP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/'
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
def s1names(pref,split):
    return pl.read_parquet(f"{pref}/p_{split}_s1.parquet",columns=['id','country','core']).select((pl.col('id').str.slice(3).cast(pl.Int64)+10_000_000_000).alias('s1'),'country',pl.col('core').list.unique().list.sort().list.join(' ').alias('ck'))
bins=[0.05,0.1,0.2,0.3,0.4,0.5,0.7]
if sys.argv[1]=='train':
    import importlib.util
    o=pl.read_parquet(R+'/work2/oof_ce4q.parquet',columns=['s1','t','y','p2'])
    acc=o.filter(pl.col('p2')>=0.7)
    emp=o.select('s1').unique().join(acc.select('s1').unique(),on='s1',how='anti')
    b=o.join(emp,on='s1',how='semi').sort('p2',descending=True).group_by('s1').head(1).join(acc.select('t').unique(),on='t',how='anti')
    tb=o.group_by('t').agg(tmax=pl.col('p2').max()); b=b.join(tb,on='t').filter(pl.col('p2')>=pl.col('tmax'))
    T=pl.concat([pl.read_parquet(f"{R}/work/p_train_s{s}.parquet",columns=['id','acore']).select((pl.col('id').str.slice(3).cast(pl.Int64)+s*10_000_000_000).alias('t'),pl.col('acore').list.len().alias('alen')) for s in (2,3)])
    b=b.join(T,on='t').filter(pl.col('alen')==0)
    S=s1names(R+'/work','train'); S=S.with_columns(mult=pl.len().over('country','ck'))
    b=b.join(S,on='s1').with_columns(bin=pl.col('p2').cut(bins),u=(pl.col('mult')==1))
    print(b.filter(pl.col('p2')>0.05).group_by('u','bin').agg(n=pl.len(),true=pl.col('y').mean().round(3)).sort('u','bin'))
    print(b.group_by('country').agg(pl.len(), (pl.col('mult')==1).mean()))
else:
    x=pl.read_parquet(SP+'emp_cls_test.parquet').filter((pl.col('acls')=='Tempty')&(pl.col('tbest')==1))
    S=pl.concat([s1names(OP+'fr6/work_unseen','test').filter(pl.col('country')=='France'),s1names(R+'/work','test').filter(pl.col('country')!='France')]).with_columns(mult=pl.len().over('country','ck'))
    x=x.join(S.select('s1','mult'),on='s1').with_columns(u=(pl.col('mult')==1))
    x.write_parquet(SP+'tempty_test.parquet')
    N=S.group_by('country').agg(N=pl.len(),uniq=(pl.col('mult')==1).mean())
    print(N)
    r=x.group_by('country','u','bin').agg(k=pl.len()).join(N,on='country').with_columns(per1k=(pl.col('k')*1000/pl.col('N')).round(3))
    print(r.pivot(on='country',index=['u','bin'],values='per1k').sort('u','bin'))
