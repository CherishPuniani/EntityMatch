"""Audit of the type-word veto on fr5: removed accepted links by added-token group; copy-count test per group
(other accepted links of the S1 after the veto; copy-like ~3.08, sibling-like ~3.21-3.26)."""
import polars as pl, sys
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)


def links(o):
    m = pl.read_csv(f"{o}/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    return (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "").select(
                (pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"),
                (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000)
                 + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t")))


a = links(f"{SP}/fr5/output_fr5_noveto"); b = links(f"{SP}/fr5/output_fr5")
rem = a.join(b, on=["s1", "t"], how="anti")
T = pl.concat([pl.read_parquet(f"{SP}/fr5/work_unseen/p_test_s{s}.parquet", columns=["id", "country", "nt", "core"]).filter(pl.col("country") == "France").select(
    (pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "nt", "core") for s in (1, 2, 3)])
rem = rem.join(T.rename({"id": "s1", "nt": "nt1", "core": "c1"}), on="s1").join(T.rename({"id": "t", "nt": "nt2", "core": "c2"}), on="t")
rem = rem.with_columns(pl.col("c2").list.set_difference("nt1").list.first().alias("a"), (pl.col("c1").list.set_difference("nt2").list.len() > 0).alias("repl"))
CITY = ["lille", "nantes", "bordeaux", "dunkerque", "tourcoing", "roubaix", "calais", "pessac", "merignac", "pornic", "saint", "nazaire", "herblain"]
TW = ["club", "amicale", "ecole", "comite", "sportive", "amis", "parents", "union", "college", "primaire", "societe", "fetes", "pharmacie",
      "anciens", "sante", "maison", "loisirs", "groupement", "lycee", "culturelle", "sport", "maternelle", "jeunes", "elementaire", "foyer",
      "culture", "section", "collectif", "institut", "patrimoine", "gestion", "danse", "musique", "residence", "ehpad", "conseil", "theatre", "centre", "service", "federation"]
rem = rem.with_columns(pl.when(pl.col("a").is_in(TW)).then(pl.lit("type word")).when(pl.col("a").is_in(CITY)).then(pl.lit("city")).otherwise(pl.lit("other")).alias("grp"))
print("removed accepted links by group / replace:"); print(rem.group_by("grp", "repl").len().sort("grp", "repl"))
print("top 'other' tokens:", rem.filter(pl.col("grp") == "other").group_by("a").len().sort("len", descending=True).head(25).rows())
oth = b.join(rem.select("s1").unique(), on="s1", how="semi").group_by("s1").len()
g = rem.select("s1", "grp").unique().join(oth, on="s1", how="left").with_columns(pl.col("len").fill_null(0)).group_by("grp").agg(
    pl.len().alias("S1s"), pl.col("len").mean().round(3).alias("mean_other_links"))
print("copy-count per group (other accepted links after the veto):"); print(g)
