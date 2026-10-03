"""Unseen-country pairs: null the raw-text dense similarity (dscore, drank) when exactly one side's raw name carries a
remapped French filler (groupe/developpement/france/compagnie/associes) - the embedding saw the unnormalised word."""
import polars as pl, unicodedata, re
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"; OUT = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/fr6/pairfeats_fr6.parquet"
seen = set(pl.read_parquet(f"{R}/work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
TOK = r"\b(groupe|developpement|france|compagnie|associes)\b"
def fold(s): return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()
recs = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(~pl.col("country").is_in(list(seen))).select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "business_name") for s in (1, 2, 3)])
recs = recs.with_columns(pl.col("business_name").map_elements(lambda s: " ".join(sorted(set(re.findall(TOK, fold(s))))), return_dtype=pl.Utf8).alias("ft")).select("id", "ft")
pf = pl.read_parquet(f"{R}/work/dense/pairfeats_test.parquet").join(recs.rename({"id": "s1", "ft": "f1"}), on="s1").join(recs.rename({"id": "t", "ft": "f2"}), on="t", how="left")
stale = pl.col("f1") != pl.col("f2").fill_null("")
print("unseen-country pairfeats rows", pf.height, "| stale (dense nulled):", pf.filter(stale & pl.col("dscore").is_not_null()).height)
pf.with_columns(pl.when(stale).then(None).otherwise(pl.col("dscore")).alias("dscore"), pl.when(stale).then(None).otherwise(pl.col("drank")).alias("drank")).drop("f1", "f2").write_parquet(OUT)
