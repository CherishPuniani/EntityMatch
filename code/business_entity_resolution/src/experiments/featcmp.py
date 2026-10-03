"""Feature distributions of same-address drop+add pairs: US test vs France test (medians), top model features."""
import polars as pl, glob, pickle
sc = pl.read_parquet("output_shift_routed/scores.parquet", columns=["s1", "t", "p2"]).filter(pl.col("p2") >= 0.05)
s1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
F = ["rs", "rrank", "bscore", "brank", "t_ns1", "t_rank", "t_brank", "cs_tset", "cs_ratio", "core_jac", "core_covA", "core_covB", "core_unB", "core_maxunB",
     "nt_covB", "nt_unB", "as_tset", "acs_tset", "addr_jac", "addr_unB", "addr_unA", "hn_eq", "len_ns1", "len_as2", "npool", "w0", "w1", "wmax"]
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "aempty_2"] + F) for d in ["work/feat_test", "work/feat_test_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
at = pl.read_parquet("work/rec_attrs_test.parquet", columns=["id", "cc_fs1", "acs_fs1", "cc_ft", "acs_ft"])
x = sc.join(s1, on="s1").join(f, on=["s1", "t"]).filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1))
x = x.join(at.rename({"id": "s1", "cc_fs1": "cc_fs1_s", "acs_fs1": "acs_fs1_s", "cc_ft": "cc_ft_s", "acs_ft": "acs_ft_s"}), on="s1").join(
    at.rename({"id": "t", "cc_fs1": "cc_fs1_t", "acs_fs1": "acs_fs1_t", "cc_ft": "cc_ft_t", "acs_ft": "acs_ft_t"}), on="t")
x = x.with_columns(pl.when(pl.col("core_unB") > 0).then(pl.lit("adds")).otherwise(pl.lit("noadd")).alias("g"))
cols = F + ["cc_fs1_s", "acs_fs1_s", "cc_ft_s", "acs_ft_s", "cc_fs1_t", "acs_fs1_t"]
out = x.filter(pl.col("country").is_in(["US", "France"])).group_by("country", "g").agg([pl.len().alias("n"), pl.col("p2").median().alias("p2")] + [pl.col(c).median().alias(c) for c in cols])
t = out.sort("g", "country").transpose(include_header=True, column_names=[f"{c}/{g}" for c, g in zip(out.sort("g", "country")["country"], out.sort("g", "country")["g"])])
pl.Config.set_tbl_rows(60); print(t)
