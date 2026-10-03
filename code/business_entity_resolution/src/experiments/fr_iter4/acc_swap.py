"""Copy-count test restricted to ACCEPTED same-address type-word swaps (v3): S1s with an accepted swap link — other
accepted links excluding it. Copy-like ~3.05 (known fillers), sibling-like ~3.19. Also the S1 name-length split."""
import polars as pl, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
import filler_train as ft
from copycount import test
from fr_more import TW, links, x
xs = x.with_columns(pl.when(pl.col("add1").is_in(TW) & (pl.col("p") >= 0.7)).then(pl.lit("__acc_swap"))
                    .when(pl.col("add1").is_in(TW)).then(pl.lit("__rej_swap"))
                    .when(pl.col("add1").is_in(["services", "cie"]) & (pl.col("p") >= 0.7)).then(pl.lit("__acc_fill"))
                    .otherwise(pl.col("add1")).alias("add1"))
test(xs, links, {"accepted type-word swap": ["__acc_swap"], "rejected type-word swap": ["__rej_swap"], "accepted known filler": ["__acc_fill"]},
     "France: accepted vs rejected swaps")
