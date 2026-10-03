"""Shared helpers for the neural (Qwen3) components: record serialization, id handling, model paths."""
import os, re
import polars as pl

QWEN_RERANKER = os.environ.get("QWEN_RERANKER", "Qwen/Qwen3-Reranker-0.6B")
QWEN_EMBED = os.environ.get("QWEN_EMBED", "Qwen/Qwen3-Embedding-0.6B")
IND_RE = re.compile(r"[ऀ-෿]")


def int_id(col, s=None):
    """S{s}-123 -> s*1e10 + 123 (same convention as ids.py / block.py)."""
    if s is None:
        return (pl.col(col).str.slice(3).cast(pl.Int64) + pl.col(col).str.slice(1, 1).cast(pl.Int64) * 10_000_000_000)
    return pl.col(col).str.slice(3).cast(pl.Int64) + s * 10_000_000_000


def records(split, sources=(1, 2, 3), translit=True, ids=None):
    """Raw records with integer ids. Adds `tl`: the Latin transliteration of a native-script name (normalization
    from norm.py's learned dictionary + rule fallback), empty otherwise. Raw text is never replaced."""
    out = []
    for s in sources:
        df = pl.read_parquet(f"work/{split}_s{s}.parquet").with_columns(int_id("entity_id", s).alias("id"))
        df = df.select("id", pl.col("business_name").alias("name"), pl.col("business_address").alias("addr"), "country")
        out.append(df)
    df = pl.concat(out)
    if ids is not None:
        df = df.join(pl.DataFrame({"id": ids}).unique(), on="id")
    if translit:
        import norm
        names = df["name"].to_list()
        tl = [norm.indic_name_to_latin(n)[0] if IND_RE.search(n) else "" for n in names]
        df = df.with_columns(pl.Series("tl", tl))
    return df


def ser(name, addr, tl=""):
    """Ditto-style compact serialization: explicit field markers, missing address kept as an explicit token
    (unknown evidence, not disagreement); transliteration appended in parentheses, raw name preserved."""
    a = addr.strip() if addr else ""
    n = name.strip()
    if tl:
        n = f"{n} ({tl})"
    return f"name: {n} ; address: {a if a else '[none]'}"
