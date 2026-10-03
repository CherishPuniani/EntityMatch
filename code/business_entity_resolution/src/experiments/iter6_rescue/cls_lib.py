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
