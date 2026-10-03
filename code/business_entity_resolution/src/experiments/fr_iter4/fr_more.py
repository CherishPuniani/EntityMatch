"""(1) acronym class (target name = one 2-3 letter token equal to the S1 core initials) copy-count test;
(2) accepted type-word swaps: which S1 token is dropped, examples."""
import polars as pl, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
import filler_train as ft
from copycount import test
R = ft.R
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
TW = ["club", "amicale", "ecole", "comite", "sportive", "amis", "parents", "union", "college", "primaire", "societe", "fetes", "pharmacie",
      "anciens", "sante", "maison", "loisirs", "groupement", "lycee", "culturelle", "sport", "maternelle", "jeunes", "elementaire", "foyer",
      "culture", "section", "collectif", "institut", "patrimoine", "gestion", "danse", "musique", "residence", "ehpad", "conseil", "theatre"]
x = ft.load("test", "France", f"{R}/output_final_v3/scores.parquet", ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], "p2n")
m = pl.read_csv(f"{R}/output_final_v3/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
links = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != "").select(
             (pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"),
             (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000)
              + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t")))
# acronym: added token (len 2-3) equals initials of S1 core
x = x.with_columns(pl.col("c1").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("init"))
acr = x.filter((pl.col("add1").str.len_chars() <= 3) & (pl.col("add1") == pl.col("init")))
other2 = x.filter((pl.col("add1").str.len_chars() == 2) & (pl.col("add1") != pl.col("init")))
print("acronym pairs:", acr.height, "accepted share:", round((acr["p"] >= 0.7).mean(), 3),
      "| 2-letter non-initial pairs:", other2.height, round((other2["p"] >= 0.7).mean(), 3))
x2 = x.with_columns(pl.when((pl.col("add1").str.len_chars() <= 3) & (pl.col("add1") == pl.col("init"))).then(pl.lit("__acr"))
                    .otherwise(pl.col("add1")).alias("add1"))
test(x2, links, {"acronym": ["__acr"], "known filler(services/cie)": ["services", "cie"], "typeword-swap": TW}, "France acronym check")
# accepted type-word swaps
sw = x.filter(pl.col("add1").is_in(TW) & pl.col("drop"))
sw = sw.with_columns(pl.col("dropped").list.first().alias("d1"))
print("\ntype-word swaps (replace):", sw.height, "accepted:", (sw["p"] >= 0.7).sum())
print("dropped token is itself a type word:", round(sw["d1"].is_in(TW + ["centre", "service"]).mean(), 3),
      "| accepted & dropped is type word:", round(sw.filter(pl.col("p") >= 0.7)["d1"].is_in(TW + ["centre", "service"]).mean(), 3))
acc = sw.filter(pl.col("p") >= 0.7)
print(acc.group_by("d1", "add1").len().sort("len", descending=True).head(15))
names = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "business_name", "business_address") for s in (1, 2, 3)])
info = {r[0]: (r[1], r[2]) for r in names.iter_rows()}
for s1, t, p in acc.sample(12, seed=2).select("s1", "t", "p").iter_rows():
    a, b = info[s1], info[t]
    print(f"p={p:.3f}  {a[0][:38]:38s} | {a[1][:55]}\n          {b[0][:38]:38s} | {b[1][:55]}")
