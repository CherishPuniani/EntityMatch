"""Evaluate frnorm4 (label-free): (1) tree scores of French pairs not touching the changed tokens are unchanged;
(2) acceptance by added-token class before (v3) / after (fr4, fr4+veto); (3) links per S1; (4) copy-count test."""
import polars as pl, sys, os
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
sys.path.insert(0, SP)
import filler_train as ft
from copycount import test
R, B = ft.R, SP + "/fr4"
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
CH = ["groupe", "developpement", "france", "centre", "service"]
# (1)
a = pl.read_parquet(f"{R}/work2/test_scores_tree_unseen.parquet", columns=["s1", "t", "p2"])
b = pl.read_parquet(f"{B}/test_scores_tree_unseen_fr4.parquet", columns=["s1", "t", "p2"]).rename({"p2": "p2b"})
d = a.join(b, on=["s1", "t"]); assert len(d) == len(a) == len(b)
T = pl.concat([pl.read_parquet(f"{B}/work_unseen/p_test_s{s}.parquet", columns=["id", "country", "nt"]).filter(pl.col("country") == "France").select(
    (pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), pl.col("nt").list.eval(pl.element().is_in(CH)).list.any().alias("f")) for s in (1, 2, 3)])
d = d.join(T.rename({"id": "s1", "f": "f1"}), on="s1").join(T.rename({"id": "t", "f": "f2"}), on="t").with_columns((pl.col("p2b") - pl.col("p2")).abs().alias("ad"))
for lab, q in [("unaffected", ~pl.col("f1") & ~pl.col("f2")), ("affected", pl.col("f1") | pl.col("f2"))]:
    z = d.filter(q)
    print(f"(1) {lab}: pairs {z.height}, max|dp2| {z['ad'].max():.4f}, share |dp2|>0.01 {(z['ad'] > 0.01).mean():.4f}, "
          f"crossing 0.7 up {((z['p2'] < 0.7) & (z['p2b'] >= 0.7)).sum()} down {((z['p2'] >= 0.7) & (z['p2b'] < 0.7)).sum()}")
# (2)
outs = [("v3", f"{R}/output_final_v3"), ("fr4", f"{B}/output_fr4_noveto")] + ([("fr4+veto", f"{B}/output_fr4")] if os.path.exists(f"{B}/output_fr4/scores.parquet") else [])
TOK = ["groupe", "developpement", "france", "services", "cie", "centre", "service", "compagnie", "associes", "club", "amicale", "ecole", "comite", "union", "sante"]
rows = {}
for lab, o in outs:
    x = ft.load("test", "France", f"{o}/scores.parquet", ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], "p2n")
    x = x.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
    rows[lab] = x.filter(pl.col("add1").is_in(TOK)).group_by("add1").agg(pl.len().alias("n"), ((pl.col("p") >= 0.7) & (pl.col("rt") == 1)).mean().round(3).alias(lab))
    if lab == outs[-1][0]:
        xl = x
t = rows["v3"]
for lab, _ in outs[1:]:
    t = t.join(rows[lab].drop("n"), on="add1")
print("(2) same-address single-added-token pairs (frnorm2 tokenisation): accepted share"); print(t.sort("n", descending=True))
# (3) + (4)
def links_of(o):
    m = pl.read_csv(f"{o}/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    return (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "").select(
                (pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"),
                (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000)
                 + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t")))
fr = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
base = links_of(f"{R}/output_final_v3")
for lab, o in outs:
    L = links_of(o)
    Lf = L.join(fr, on="s1", how="semi")
    nonfr_same = L.join(fr, on="s1", how="anti").sort("s1", "t").equals(base.join(fr, on="s1", how="anti").sort("s1", "t"))
    print(f"(3) {lab}: French links {Lf.height} ({Lf.height / fr.height:.3f}/S1), empty {1 - Lf['s1'].n_unique() / fr.height:.4f}; "
          f"added vs v3 {Lf.join(base, on=['s1', 't'], how='anti').height}, removed {base.join(fr, on='s1', how='semi').join(Lf, on=['s1', 't'], how='anti').height}; "
          f"US/India identical to v3: {nonfr_same}")
L = links_of(outs[-1][1]).join(fr, on="s1", how="semi")
TW = ["club", "amicale", "ecole", "comite", "sportive", "amis", "parents", "union", "college", "primaire", "societe", "fetes", "pharmacie",
      "anciens", "sante", "maison", "loisirs", "groupement", "lycee", "culturelle", "sport", "maternelle", "jeunes", "elementaire", "foyer",
      "culture", "section", "collectif", "institut", "patrimoine", "gestion", "danse", "musique", "residence", "ehpad", "conseil", "theatre"]
test(xl, L, {"known filler(services/cie)": ["services", "cie"], "groupe": ["groupe"], "developpement": ["developpement"], "france": ["france"],
             "typeword-swap": TW, "centre": ["centre"], "service": ["service"]}, f"(4) copy-count test on {outs[-1][0]}")
