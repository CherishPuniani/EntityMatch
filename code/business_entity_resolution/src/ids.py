import polars as pl
def to_int(c):
    return (pl.col(c).str.slice(3).cast(pl.Int64) + pl.col(c).str.slice(1,1).cast(pl.Int64) * 10_000_000_000).alias(c)
def to_str(c):
    return pl.concat_str([pl.lit("S"), (pl.col(c) // 10_000_000_000).cast(pl.Utf8), pl.lit("-"),
                          (pl.col(c) % 10_000_000_000).cast(pl.Utf8)]).alias(c)
