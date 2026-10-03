import polars as pl, sys
split=sys.argv[1]; keep=sys.argv[2] if len(sys.argv)>2 else None; tag=sys.argv[3] if len(sys.argv)>3 else split
def load(s):
    df=pl.read_parquet(f"work/p_{split}_s{s}.parquet",columns=["id","country","nt","core","acore"])
    df=df.with_columns((pl.col("id").str.slice(3).cast(pl.Int64)+s*10_000_000_000).alias("id"))
    nums=pl.col("acore").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")))
    return df.select("id","country",
        pl.concat_str(["country",pl.when(pl.col("core").list.len()>0).then(pl.col("core")).otherwise(pl.col("nt")).list.join("")]).hash().alias("cc_h"),
        pl.when(pl.col("acore").list.len()>0).then(pl.concat_str(["country",pl.col("acore").list.join(" ")]).hash()).alias("acs_h"),
        nums.list.first().hash().alias("hn_h"),
        pl.when(nums.list.len()>0).then(nums.list.sort().list.join(" ").hash()).alias("nums_h"))
s1=load(1); t=pl.concat([load(2),load(3)])
if keep: s1=s1.join(pl.read_parquet(keep),on="id")
f1c=s1.group_by("cc_h").len("cc_fs1"); f1a=s1.drop_nulls("acs_h").group_by("acs_h").len("acs_fs1")
ftc=t.group_by("cc_h").len("cc_ft"); fta=t.drop_nulls("acs_h").group_by("acs_h").len("acs_ft")
out=pl.concat([s1,t]).join(f1c,on="cc_h",how="left").join(f1a,on="acs_h",how="left").join(ftc,on="cc_h",how="left").join(fta,on="acs_h",how="left")
out=out.with_columns([pl.col(c).fill_null(0).cast(pl.Int32) for c in ["cc_fs1","acs_fs1","cc_ft","acs_ft"]]).drop("country")
out.write_parquet(f"work/rec_attrs_{tag}.parquet"); print(out.shape, out.describe())
