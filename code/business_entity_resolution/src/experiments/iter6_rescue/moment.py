import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
R='/home2/home/amritanshu_t/amazon-mlc-26/run'
o=pl.read_parquet(R+'/work2/oof_ce4q.parquet',columns=['s1','t','y','p2'])
ctr=pl.read_parquet(R+'/work/p_train_s1.parquet',columns=['id','country']).select((pl.col('id').str.slice(3).cast(pl.Int64)+10_000_000_000).alias('s1'),'country')
keep=o.select('s1').unique().join(ctr,on='s1')
gt=pl.read_parquet(R+'/work/gt_pairs_int.parquet').join(keep,on='s1',how='semi')
# approx OOF decisions: p2>=0.7 and s1 is t's argmax
o=o.with_columns(tmax=pl.col('p2').max().over('t'))
acc=o.filter((pl.col('p2')>=0.7)&(pl.col('p2')>=pl.col('tmax')))
def dist(links,name):
    n=keep.join(links.group_by('s1').agg(n=pl.len()),on='s1',how='left').with_columns(pl.col('n').fill_null(0))
    return n.group_by('country').agg(empty=(pl.col('n')==0).mean().round(5),mean=pl.col('n').mean().round(4)).with_columns(pl.lit(name).alias('src'))
print(pl.concat([dist(gt,'truth'),dist(acc,'pred_oof')]).sort('country'))
tp=acc.filter(pl.col('y')==1).join(keep,on='s1').group_by('country').agg(tp=pl.len()); fp=acc.filter(pl.col('y')==0).join(keep,on='s1').group_by('country').agg(fp=pl.len())
N=keep.group_by('country').agg(N=pl.len()); T=gt.join(keep,on='s1').group_by('country').agg(T=pl.len())
print(N.join(tp,on='country').join(fp,on='country').join(T,on='country').with_columns(fp_per=pl.col('fp')/pl.col('N'),fn_per=(pl.col('T')-pl.col('tp'))/pl.col('N')))
