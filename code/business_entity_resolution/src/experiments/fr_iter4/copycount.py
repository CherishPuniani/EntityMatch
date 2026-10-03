"""Copy-count test. The generator draws a number of copies per S1; if a class-C candidate (same address, one added token)
is a copy, S1s having one show ~1 fewer OTHER links than S1s without; if C is a sibling, no deficit.
Train (labels: links = true copies) calibrates the test on known fillers / siblings; test France applies it
(links = accepted links of the final v3)."""
import polars as pl, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
import filler_train as ft
R = ft.R
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)


def test(x, links, classes, lab):
    """x: single-added-token same-address pairs (s1, t, add1); links: (s1, t) positive links; classes: {name: tokens}."""
    base_n = links.group_by("s1").len()
    rows = []
    for cname, toks in classes.items():
        c = x.filter(pl.col("add1").is_in(toks)).select("s1", "t").unique()
        s1c = c.select("s1").unique()
        other = links.join(c, on=["s1", "t"], how="anti").join(s1c, on="s1").group_by("s1").len()
        v = s1c.join(other, on="s1", how="left").with_columns(pl.col("len").fill_null(0))
        linked_c = links.join(c, on=["s1", "t"], how="semi").height
        rows.append((cname, s1c.height, round(v["len"].mean(), 3), round(linked_c / max(c.height, 1), 3)))
    allc = x.select("s1").unique()
    nonc = base_n.join(allc, on="s1", how="anti")
    print(f"\n== {lab}: mean links per S1 among S1s with NO same-address single-token candidate: {nonc['len'].mean():.3f}"
          f" (S1s with >=1 link: {nonc.height})")
    print(pl.DataFrame(rows, schema=["class", "S1s", "mean_other_links", "share_C_linked"], orient="row"))


if __name__ == "__main__":
    for c, cls in [("India", {"filler(center/services/service/partners)": ["center", "services", "service", "partners"],
                              "descriptor(enterprises..group)": ["enterprises", "industries", "ventures", "exports", "overseas", "infratech", "holdings", "group", "public"],
                              "typeword-swap(international..)": ["international", "global", "consultants", "developers", "properties", "management", "foundation",
                                                                 "marketing", "agro", "infra", "infrastructure", "finance", "producer", "healthcare", "india"],
                              "legal(ltd/limited/co..)": ["ltd", "limited", "company", "corporation", "co", "corp"]}),
                   ("US", {"filler(center/services/service/partners)": ["center", "services", "service", "partners"],
                           "descriptor(holdings/group)": ["holdings", "group"], "legal(inc/llc/..)": ["inc", "llc", "corp", "co", "ltd", "lp"]})]:
        x = ft.load("train", c, f"{R}/work2/oof_ce4q.parquet", ["work/feat_trainD", "work/feat_trainD_dense"], "p2")
        links = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "y"]).filter(pl.col("y") == 1).select("s1", "t").join(
            x.select("s1").unique().vstack(pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1"]).unique()).unique(), on="s1", how="semi")
        cs1 = pl.read_parquet(f"{R}/work/p_train_s1.parquet", columns=["id", "country"]).filter(pl.col("country") == c).select(
            (pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
        test(x, links.join(cs1, on="s1", how="semi"), cls, f"TRAIN {c} (true copies)")
    x = ft.load("test", "France", f"{R}/output_final_v3/scores.parquet", ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], "p2n")
    m = pl.read_csv(f"{R}/output_final_v3/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    links = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "").select(
                 (pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"),
                 (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000)
                  + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t")))
    fr1 = x.select("s1").unique()
    frs1 = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(
        (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
    TW = ["club", "amicale", "ecole", "comite", "sportive", "amis", "parents", "union", "college", "primaire", "societe", "fetes", "pharmacie",
          "anciens", "sante", "maison", "loisirs", "groupement", "lycee", "culturelle", "sport", "maternelle", "jeunes", "elementaire", "foyer",
          "culture", "section", "collectif", "institut", "patrimoine", "gestion", "danse", "musique", "residence", "ehpad", "conseil", "theatre"]
    test(x, links.join(frs1, on="s1", how="semi"),
         {"known filler(services/cie)": ["services", "cie"], "groupe": ["groupe"], "developpement": ["developpement"], "france": ["france"],
          "typeword-swap(club..theatre)": TW, "centre": ["centre"], "service": ["service"], "compagnie": ["compagnie"], "associes": ["associes"],
          "federation": ["federation"], "legal(sa/sci/sas/sasu/eurl)": ["sa", "sci", "sas", "sasu", "eurl"], "sarl": ["sarl"]},
         "TEST France (accepted links, final v3)")
