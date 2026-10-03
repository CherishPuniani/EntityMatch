"""Expected-F0.5 decoding per S1. Eligible candidates = pairs where the S1 is the target's best (exclusivity).
For each S1, choose the top-m eligible candidates (by p) maximising E[F0.5] under independent Bernoulli truths:
  F = 1.25 A / (1.25 A + 0.25 B + (m - A)),  A = true among chosen, B = true among the rest; m = 0: F = 1{B = 0}.
S1s whose eligible p are all outside [LO, HI] keep the threshold decision (identical by construction for p far from 0.5-0.8).
decode(df[s1, t, p]) -> df[s1, t] selected."""
import numpy as np, polars as pl, os
LO = float(os.environ.get("EF_LO", "0.3")); HI = float(os.environ.get("EF_HI", "0.95")); PMIN = float(os.environ.get("EF_PMIN", "0.02"))
TH = 0.7


def _pb(ps):
    d = np.zeros(len(ps) + 1); d[0] = 1.0
    for i, p in enumerate(ps):
        d[1:i + 2] = d[1:i + 2] * (1 - p) + d[0:i + 1] * p
        d[0] *= (1 - p)
    return d


def _best_m(ps, w):
    """ps sorted desc; w = scale (1.0 = trust p). returns best m."""
    ps = np.clip(np.asarray(ps) * w, 0, 1)
    n = len(ps)
    # suffix distributions of B
    suf = [None] * (n + 1); suf[n] = np.array([1.0])
    for j in range(n - 1, -1, -1):
        prev = suf[j + 1]; d = np.zeros(len(prev) + 1)
        d[:-1] += prev * (1 - ps[j]); d[1:] += prev * ps[j]; suf[j] = d
    best, bm = suf[0][0], 0  # m = 0
    A = np.array([1.0])
    for m in range(1, n + 1):
        p = ps[m - 1]; d = np.zeros(len(A) + 1); d[:-1] += A * (1 - p); d[1:] += A * p; A = d
        B = suf[m]
        a = np.arange(len(A))[:, None]; b = np.arange(len(B))[None, :]
        F = np.where(a > 0, 1.25 * a / (1.25 * a + 0.25 * b + (m - a)), 0.0)
        e = float((A[:, None] * B[None, :] * F).sum())
        if e > best + 1e-12:
            best, bm = e, m
    return bm


def decode(df, w=1.0):
    df = df.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
    el = df.filter((pl.col("rt") == 1) & (pl.col("p") >= PMIN))
    amb = el.filter((pl.col("p") >= LO) & (pl.col("p") < HI)).select("s1").unique()
    base = el.filter(pl.col("p") >= TH).join(amb, on="s1", how="anti").select("s1", "t")
    g = el.join(amb, on="s1").sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).agg("t", "p")
    sel_s, sel_t = [], []
    for s, ts, ps in g.iter_rows():
        m = _best_m(ps, w)
        sel_s += [s] * m; sel_t += ts[:m]
    new = pl.DataFrame({"s1": sel_s, "t": sel_t}, schema={"s1": pl.Int64, "t": pl.Int64})
    return pl.concat([base, new])
