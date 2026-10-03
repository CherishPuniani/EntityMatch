"""Extended retrieval: pairs (S1, target) sharing country + house number + a rare (pool<=POOL) address token, NOT in the
existing candidate set; cheap features. usage: xr_build.py train|test  -> xr_<split>.parquet"""
import polars as pl, numpy as np, sys, os
from rapidfuzz import process, fuzz
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
split = sys.argv[1]; POOL = int(os.environ.get("POOL", "15"))
prep = {"train": f"{R}/work", "test": f"{SP}/fr6/work_unseen"}[split]   # test: French records carry the fr6 normaliser
P = pl.concat([pl.read_parquet(f"{prep}/p_{split}_s{s}.parquet", columns=["id", "country", "nf", "core", "acore"]).select(
    (pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "country", "nf",
    pl.col("core").list.join(" ").alias("cs"), pl.col("core").alias("core"), pl.col("acore").list.unique().alias("aset"),
    pl.col("acore").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first().alias("hn"),
    pl.col("acore").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$") & (pl.element().str.len_chars() >= 3))).list.unique().alias("toks")) for s in (1, 2, 3)])
if split == "train":
    keep = pl.read_parquet(f"{R}/work/keep_trainD.parquet").rename({"id": "s1"})
    base = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "p2"]).rename({"p2": "p"})
    S1ids = keep
else:
    base = pl.read_parquet(f"{SP}/final7/output_final7/scores.parquet", columns=["s1", "t", "p2n"]).rename({"p2n": "p"})
    S1ids = P.filter(pl.col("id") < 20_000_000_000).select(pl.col("id").alias("s1"))
T = P.filter((pl.col("id") >= 20_000_000_000) & pl.col("hn").is_not_null()).select("id", "country", "hn", "toks").explode("toks").drop_nulls("toks")
pool = T.group_by("country", "hn", "toks").len("pool").filter(pl.col("pool") <= POOL)
S = P.filter(pl.col("id") < 20_000_000_000).rename({"id": "s1"}).join(S1ids, on="s1").filter(pl.col("hn").is_not_null()).select("s1", "country", "hn", "toks").explode("toks").drop_nulls("toks")
x = S.join(pool, on=["country", "hn", "toks"]).join(T.rename({"id": "t"}), on=["country", "hn", "toks"])
x = x.group_by("s1", "t").agg(pl.len().alias("nkey"), pl.col("pool").min().alias("minpool"))
x = x.join(base.select("s1", "t"), on=["s1", "t"], how="anti")
print(split, "extended pairs:", x.height, flush=True)
# context from the base output: is the target already claimed, S1's accepted count and best p
bs = base.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
claim = bs.filter(pl.col("rt") == 1).select("t", pl.col("p").alias("t_bestp"))
s1acc = bs.filter((pl.col("p") >= 0.7) & (pl.col("rt") == 1)).group_by("s1").len("s1_nacc")
x = x.join(claim, on="t", how="left").join(s1acc, on="s1", how="left").with_columns(pl.col("t_bestp").fill_null(0.0), pl.col("s1_nacc").fill_null(0))
A = P.select(pl.col("id").alias("s1"), "country", pl.col("nf").alias("nf1"), pl.col("cs").alias("cs1"), pl.col("core").alias("c1"), pl.col("aset").alias("a1"))
B = P.select(pl.col("id").alias("t"), pl.col("nf").alias("nf2"), pl.col("cs").alias("cs2"), pl.col("core").alias("c2"), pl.col("aset").alias("a2"))
x = x.join(A, on="s1").join(B, on="t")
n1, n2, c1, c2 = (x[c].fill_null("").to_list() for c in ["nf1", "nf2", "cs1", "cs2"])
x = x.with_columns(pl.Series("ns_tset", process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=24).astype(np.float32)),
                   pl.Series("ns_ratio", process.cpdist(n1, n2, scorer=fuzz.ratio, workers=24).astype(np.float32)),
                   pl.Series("cs_tset", process.cpdist(c1, c2, scorer=fuzz.token_set_ratio, workers=24).astype(np.float32)),
                   pl.Series("cs_part", process.cpdist(c1, c2, scorer=fuzz.partial_ratio, workers=24).astype(np.float32)))
x = x.with_columns(pl.col("c1").list.set_intersection("c2").list.len().alias("core_inter"), pl.col("c1").list.len().alias("core_n1"), pl.col("c2").list.len().alias("core_n2"),
                   pl.col("a1").list.set_intersection("a2").list.len().alias("addr_inter"), pl.col("a1").list.len().alias("addr_n1"), pl.col("a2").list.len().alias("addr_n2"),
                   pl.col("nf2").str.len_chars().alias("len2"), pl.col("nf2").str.contains(r"\.com|www|@|#").alias("web2"),
                   pl.col("country").replace_strict({"US": 0, "India": 1, "France": 2}, default=3).alias("cty"))
x = x.with_columns((pl.col("addr_inter") / pl.max_horizontal(pl.col("addr_n1"), 1)).alias("addr_cov1"), (pl.col("addr_inter") / pl.max_horizontal(pl.col("addr_n2"), 1)).alias("addr_cov2"))
x = x.drop("nf1", "nf2", "cs1", "cs2", "c1", "c2", "a1", "a2")
if split == "train":
    gt = pl.read_parquet(f"{R}/work/gt_pairs_int.parquet").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    x = x.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    x = x.join(pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "fold"]).unique("s1"), on="s1", how="left")
    print("positives:", x["y"].sum(), flush=True)
x.write_parquet(f"{SP}/xr/xr_{split}.parquet"); print("wrote", x.shape)
