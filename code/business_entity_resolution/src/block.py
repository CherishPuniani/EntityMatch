"""Candidate generation via compound hashed keys + IDF-weighted overlap."""
import polars as pl, numpy as np, time, sys

# rarest name tokens / addr alpha tokens / addr numbers per record, (S1 side, target side)
NT, AT, HT = 3, 4, 2
PARAMS = {"s1": (4, 7, 3), "t": (4, 5, 2)}
TYPES = {"N1": 0, "AA": 1, "NP": 2, "NA": 3, "NN": 4, "HA": 5, "NH": 6}  # anything else ("C...") -> 7
NTY = 8
BLOCK_STOP = set("com www id dba formerly aka fka nee doing as known f k a d b the and of".split())


def base(split, s):
    import norm
    df = pl.read_parquet(f"work/p_{split}_s{s}.parquet", columns=["id", "country", "nt", "core", "acore"])
    df = df.with_columns((pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"))
    stop = list(norm.LEGAL | norm.HONOR | BLOCK_STOP)
    df = df.with_columns(pl.col("nt").list.eval(pl.element().filter(~pl.element().is_in(stop))).alias("bt"))
    df = df.with_columns(
        pl.when(pl.col("core").list.len() > 0).then(pl.col("core")).otherwise(pl.col("nt")).alias("ntok"),
        pl.when(pl.col("bt").list.len() > 0).then(pl.col("bt")).otherwise(pl.col("nt")).alias("bt"),
        (pl.col("acore").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.len() == 0).alias("noaddr"))
    return df


def token_tables(df, tdf, side):
    """rarest tokens per record, rarity from target-side df tables."""
    NT, AT, HT = PARAMS[side]
    n = (df.select("id", "country", pl.col("bt").list.unique().alias("t")).explode("t")
         .filter(pl.col("t").str.len_chars() >= 2)
         .join(tdf["n"], on=["country", "t"], how="left").with_columns(pl.col("df").fill_null(0))
         .sort(["id", "df"]).group_by("id", maintain_order=True).head(NT))
    a = (df.select("id", "country", pl.col("acore").list.unique().alias("t")).explode("t")
         .filter(pl.col("t").str.len_chars() >= 3, ~pl.col("t").str.contains(r"^\d+$"))
         .join(tdf["a"], on=["country", "t"], how="left").with_columns(pl.col("df").fill_null(0))
         .sort(["id", "df"]).group_by("id", maintain_order=True).head(AT))
    h = (df.select("id", "country", pl.col("acore").list.unique().alias("t")).explode("t")
         .filter(pl.col("t").str.contains(r"^\d+$"), pl.col("t") != "0")
         .join(tdf["h"], on=["country", "t"], how="left").with_columns(pl.col("df").fill_null(0))
         .sort(["id", "df"]).group_by("id", maintain_order=True).head(HT))
    return n.select("id", "country", "t"), a.select("id", "t"), h.select("id", "t")


def keys(df, tdf, side):
    n, a, h = token_tables(df, tdf, side)
    parts = []
    # single name token keys: target side only when it has no usable address
    n1 = n if side == "s1" else n.join(df.filter(pl.col("noaddr")).select("id"), on="id")
    parts.append(n1.select("id", "country", pl.concat_str([pl.lit("N1"), "t"]).alias("k")))
    # address token pairs (renamed businesses)
    aa = a.join(a, on="id", suffix="2").filter(pl.col("t") < pl.col("t2")).join(
        df.select("id", "country"), on="id")
    parts.append(aa.select("id", "country", pl.concat_str([pl.lit("AA"), "t", pl.lit("|"), "t2"]).alias("k")))
    # typo-tolerant name-pair key on 4-char prefixes
    npf = n.join(n.select("id", "t"), on="id", suffix="2").filter(pl.col("t") < pl.col("t2"))
    parts.append(npf.select("id", "country", pl.concat_str([pl.lit("NP"), pl.col("t").str.slice(0, 4), pl.lit("|"),
                                                            pl.col("t2").str.slice(0, 4)]).alias("k")))
    na = n.join(a, on="id", suffix="2")
    parts.append(na.select("id", "country", pl.concat_str([pl.lit("NA"), "t", pl.lit("|"), "t2"]).alias("k")))
    nn = n.join(n.select("id", "t"), on="id", suffix="2").filter(pl.col("t") < pl.col("t2"))
    parts.append(nn.select("id", "country", pl.concat_str([pl.lit("NN"), "t", pl.lit("|"), "t2"]).alias("k")))
    ha = h.join(a, on="id", suffix="2").join(df.select("id", "country"), on="id")
    parts.append(ha.select("id", "country", pl.concat_str([pl.lit("HA"), "t", pl.lit("|"), "t2"]).alias("k")))
    nh = n.join(h, on="id", suffix="2")
    parts.append(nh.select("id", "country", pl.concat_str([pl.lit("NH"), "t", pl.lit("|"), "t2"]).alias("k")))
    c = df.select("id", "country", pl.concat_str([pl.lit("C"), pl.col("ntok").list.join("")]).alias("k"))
    c2 = df.select("id", "country", pl.concat_str([pl.lit("C"), pl.col("nt").list.join("")]).alias("k"))
    parts += [c, c2]
    k = pl.concat(parts).drop_nulls().with_columns(
        pl.concat_str(["country", "k"]).hash().alias("h"),
        pl.col("k").str.slice(0, 2).alias("ty")).select("id", "h", "ty").unique(["id", "h"])
    k = k.with_columns(pl.col("ty").replace_strict(TYPES, default=len(TYPES)).cast(pl.Int8))
    return k


def df_tables(tgt):
    tdf = {}
    tdf["n"] = (tgt.select("country", pl.col("bt").list.unique().alias("t")).explode("t")
                .group_by("country", "t").len("df"))
    ex = tgt.select("country", pl.col("acore").list.unique().alias("t")).explode("t").group_by("country", "t").len("df")
    tdf["a"] = ex
    tdf["h"] = ex
    return tdf


def run(split, cap=300, K=30, s1_ids=None, log=print, chunk=1_000_000, s1_chunk=250_000, ranker="work/ranker.txt"):
    import lightgbm as lgb
    rk = lgb.Booster(model_file=ranker); RF = rk.feature_name()
    t = time.time()
    tgt = pl.concat([base(split, 2), base(split, 3)])
    tdf = df_tables(tgt)
    N = len(tgt)
    noaddr = tgt.select(pl.col("id").alias("id_t"), pl.col("noaddr").cast(pl.Int8))
    kt = pl.concat([keys(tgt.slice(i, chunk), tdf, "t") for i in range(0, N, chunk)])
    del tgt
    kdf = kt.group_by("h").len("df").filter(pl.col("df") <= cap)
    kdf = kdf.with_columns((np.log(N) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32).alias("w"))
    kt = kt.drop("ty").join(kdf.select("h", "w"), on="h")
    del kdf
    log(f"target keys after cap {len(kt)} {time.time()-t:.0f}s")
    s1 = base(split, 1)
    if s1_ids is not None:
        s1 = s1.filter(pl.col("id").is_in(s1_ids.implode()))
    out = []
    agg = [pl.col("w").sum().alias("bscore"), pl.len().cast(pl.Int16).alias("nkeys"), pl.col("w").max().alias("wmax")]
    agg += [pl.col("w").filter(pl.col("ty") == k).sum().alias(f"w{k}") for k in range(NTY)]
    for i in range(0, len(s1), s1_chunk):
        ks = keys(s1.slice(i, s1_chunk), tdf, "s1")
        pairs = ks.join(kt, on="h", suffix="_t").group_by("id", "id_t").agg(agg).join(noaddr, on="id_t", how="left")
        pairs = pairs.with_columns(
            pl.col("bscore").rank("ordinal", descending=True).over("id").cast(pl.Int32).alias("brank"),
            pl.len().over("id").cast(pl.Int32).alias("npool"),
            (pl.col("bscore") / pl.col("bscore").max().over("id")).alias("brel"))
        pairs = pairs.with_columns(pl.Series("rs", rk.predict(pairs.select(RF).to_numpy().astype(np.float32),
                                                              num_threads=30).astype(np.float32)))
        pairs = (pairs.with_columns(pl.col("rs").rank("ordinal", descending=True).over("id").cast(pl.Int16).alias("rrank"))
                 .filter(pl.col("rrank") <= K))
        out.append(pairs)
        log(f"s1 chunk {i} pairs {len(pairs)} {time.time()-t:.0f}s")
    del kt
    pairs = pl.concat(out)
    # target-side competition among S1s (ranker score and idf score)
    pairs = pairs.with_columns(
        pl.len().over("id_t").cast(pl.Int16).alias("t_ns1"),
        pl.col("rs").max().over("id_t").alias("t_best"),
        pl.col("rs").rank("ordinal", descending=True).over("id_t").cast(pl.Int16).alias("t_rank"),
        pl.col("bscore").max().over("id_t").alias("t_bbest"),
        pl.col("bscore").rank("ordinal", descending=True).over("id_t").cast(pl.Int16).alias("t_brank"))
    log(f"topK pairs {len(pairs)} {time.time()-t:.0f}s")
    return pairs.rename({"id": "s1", "id_t": "t"})


def to_str_ids(df, cols=("s1", "t")):
    out = []
    for c in cols:
        src = (pl.col(c) // 10_000_000_000)
        out.append(pl.concat_str([pl.lit("S"), src.cast(pl.Utf8), pl.lit("-"),
                                  (pl.col(c) % 10_000_000_000).cast(pl.Utf8)]).alias(c))
    return df.with_columns(out)
