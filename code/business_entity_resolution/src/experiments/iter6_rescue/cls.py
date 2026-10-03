import polars as pl, sys
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
R='/home2/home/amritanshu_t/amazon-mlc-26/run'
OP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/'
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
def prep(pref,split):
    return pl.concat([pl.read_parquet(f"{pref}/p_{split}_s{s}.parquet", columns=["id","country","acore","core"]).select(
        (pl.col("id").str.slice(3).cast(pl.Int64) + s*10_000_000_000).alias("id"),"country",
        pl.col("acore").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first().alias("hn"),
        pl.col("acore").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).alias("st"),
        pl.col("acore").list.len().alias("alen"),
        pl.col("core").list.unique().list.sort().list.join(" ").alias("ck")) for s in (1,2,3)])
def classify(x,P):
    A=P.select(pl.col('id').alias('s1'),pl.col('hn').alias('hn1'),pl.col('st').alias('st1'),pl.col('ck').alias('ck1'))
    B=P.select(pl.col('id').alias('t'),pl.col('hn').alias('hn2'),pl.col('st').alias('st2'),'alen',pl.col('ck').alias('ck2'))
    x=x.join(A,on='s1').join(B,on='t')
    stov=pl.col('st1').list.set_intersection('st2').list.len()/pl.max_horizontal(pl.col('st1').list.len(),1)
    return x.with_columns(acls=pl.when(pl.col('alen')==0).then(pl.lit('Tempty'))
        .when((pl.col('hn1')==pl.col('hn2'))&(stov>=0.5)).then(pl.lit('same'))
        .when(pl.col('hn1').is_null()|pl.col('hn2').is_null()).then(pl.lit('nonum'))
        .otherwise(pl.lit('diff')), neq=(pl.col('ck1')==pl.col('ck2')).cast(pl.Int8))
bins=[0.05,0.1,0.2,0.3,0.4,0.5,0.7]
if sys.argv[1]=='test':
    b=pl.read_parquet(SP+'emp_best_test.parquet').filter(pl.col('cl')==0)
    P=pl.concat([prep(OP+'fr6/work_unseen','test').filter(pl.col('country')=='France'),prep(R+'/work','test').filter(pl.col('country')!='France')])
    x=classify(b,P).with_columns(bin=pl.col('p2n').cut(bins))
    x.write_parquet(SP+'emp_cls_test.parquet')
    N=pl.read_parquet('work/p_test_s1.parquet',columns=['country']).group_by('country').agg(N=pl.len())
    r=x.group_by('country','acls','bin').agg(k=pl.len()).join(N,on='country').with_columns(per1k=(pl.col('k')*1000/pl.col('N')).round(2))
    print(r.pivot(on='country',index=['acls','bin'],values='per1k').sort('acls','bin'))
else:
    o=pl.read_parquet(R+'/work2/oof_ce4q.parquet',columns=['s1','t','y','p2'])
    acc=o.filter(pl.col('p2')>=0.7)
    emp=o.select('s1').unique().join(acc.select('s1').unique(),on='s1',how='anti')
    claimed=acc.select('t').unique()
    b=o.join(emp,on='s1',how='semi').sort('p2',descending=True).group_by('s1').head(1).join(claimed,on='t',how='anti')
    gt=pl.read_parquet(R+'/work/gt_pairs_int.parquet')
    ns=gt.group_by('s1').agg(ntrue=pl.len())
    b=b.join(ns,on='s1',how='left').with_columns(pl.col('ntrue').fill_null(0))
    P=prep(R+'/work','train')
    x=classify(b,P).with_columns(bin=pl.col('p2').cut(bins))
    print('OOF approx-empty S1 (max p2<0.7), orphan best cand: n, true rate, singleton rate, mean ntrue|true')
    print(x.filter(pl.col('p2')>0.05).group_by('acls','bin').agg(n=pl.len(),true=pl.col('y').mean().round(3),single=(pl.col('ntrue')==0).mean().round(3),ntrue_if_true=pl.col('ntrue').filter(pl.col('y')==1).mean().round(2)).sort('acls','bin'))
    print(x.filter(pl.col('p2')>0.05).group_by('acls','neq').agg(n=pl.len(),true=pl.col('y').mean().round(3)).sort('acls','neq'))
