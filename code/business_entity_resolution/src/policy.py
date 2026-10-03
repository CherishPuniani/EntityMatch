"""Decision policies turning per-pair probabilities into per-S1 match sets."""
import numpy as np, polars as pl
from numba import njit


@njit(cache=True)
def _pb(ps):
    """Poisson-binomial pmf of sum of Bernoulli(ps)."""
    d = np.zeros(len(ps) + 1)
    d[0] = 1.0
    for i in range(len(ps)):
        p = ps[i]
        for j in range(i + 1, 0, -1):
            d[j] = d[j] * (1 - p) + d[j - 1] * p
        d[0] *= (1 - p)
    return d


@njit(cache=True)
def best_k(ps, mu, maxm):
    """ps sorted desc. Returns k maximizing E[5TP/(G+4k)] (k=0 -> E=P(G=0))."""
    n = len(ps)
    # distribution of missed matches M ~ Poisson(mu) truncated
    pm = np.zeros(maxm + 1)
    for m in range(maxm + 1):
        f = 1.0
        for j in range(1, m + 1):
            f *= mu / j
        pm[m] = np.exp(-mu) * f
    best = -1.0; bk = 0
    for k in range(0, n + 1):
        dt = _pb(ps[:k])
        dr = _pb(ps[k:])
        e = 0.0
        if k == 0:
            e = dr[0] * pm[0]
        else:
            for tp in range(1, k + 1):
                if dt[tp] < 1e-12:
                    continue
                for r in range(0, n - k + 1):
                    if dr[r] < 1e-12:
                        continue
                    for m in range(maxm + 1):
                        g = tp + r + m
                        e += dt[tp] * dr[r] * pm[m] * 5.0 * tp / (g + 4.0 * k)
        if e > best:
            best = e; bk = k
    return bk, best


def apply_policy(df, kind, **kw):
    """df: s1, t, p (only candidates). Returns dict s1 -> list of t."""
    df = df.sort(["s1", "p"], descending=[False, True])
    pred = {}
    if kind == "thresh":
        sel = df.filter(pl.col("p") >= kw["t"])
        for s1, ts in sel.group_by("s1").agg("t").iter_rows():
            pred[s1] = ts
        return pred
    if kind == "top_rel":
        # keep candidates with p>=t, only if top >= t0; also require p >= rel * top
        d = df.with_columns(pl.col("p").max().over("s1").alias("pmax"))
        sel = d.filter((pl.col("pmax") >= kw["t0"]) & (pl.col("p") >= kw["t"]) & (pl.col("p") >= kw["rel"] * pl.col("pmax")))
        for s1, ts in sel.group_by("s1").agg("t").iter_rows():
            pred[s1] = ts
        return pred
    if kind == "expf":
        mu = kw.get("mu", 0.1); lo = kw.get("lo", 0.005)
        d = df.filter(pl.col("p") >= lo).group_by("s1", maintain_order=True).agg("t", "p")
        for s1, ts, ps in d.iter_rows():
            ps = np.clip(np.asarray(ps, dtype=np.float64), 0, 1)
            if len(ps) > 25:
                ps = ps[:25]; ts = ts[:25]
            k, _ = best_k(ps, mu, 3)
            if k > 0:
                pred[s1] = ts[:k]
        return pred
    raise ValueError(kind)
