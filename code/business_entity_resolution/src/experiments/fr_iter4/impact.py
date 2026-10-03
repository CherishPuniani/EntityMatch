"""Bounded France impact of a new output vs v3 (per-S1 F0.5, beta=0.5), label-free:
  G_add / L_add: gain if the added links are all true / loss if all false (other decisions of v3 taken as truth);
  G_rem / L_rem: gain if the removed links were all false / loss if they were all true.
Expected LB delta = 0.1497 * (p_add*G_add - (1-p_add)*L_add + p_rem*G_rem - (1-p_rem)*L_rem)."""
import polars as pl, sys
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
NEW = sys.argv[1]


def links(o):
    m = pl.read_csv(f"{o}/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    return (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("t")))


fr = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
a = links(f"{R}/output_final_v3").join(fr, on="s1", how="semi"); b = links(NEW).join(fr, on="s1", how="semi")
add = b.join(a, on=["s1", "t"], how="anti"); rem = a.join(b, on=["s1", "t"], how="anti"); keep = a.join(b, on=["s1", "t"], how="semi")
k = keep.group_by("s1").len().rename({"len": "k"})
s = fr.join(k, on="s1", how="left").join(add.group_by("s1").len().rename({"len": "na"}), on="s1", how="left").join(
    rem.group_by("s1").len().rename({"len": "nr"}), on="s1", how="left").fill_null(0).filter((pl.col("na") > 0) | (pl.col("nr") > 0))


def f05(tp, fp, fn):
    return pl.when(tp + fp + fn == 0).then(1.0).otherwise(1.25 * tp / (1.25 * tp + 0.25 * fn + fp))


k_, na, nr = pl.col("k"), pl.col("na"), pl.col("nr")
N = fr.height
# additions: truth = keep (+ adds if true); old prediction = keep (+ rem assumed false for this term) -> isolate by ignoring rem
G_add = s.select((f05(k_ + na, 0, 0) - f05(k_, 0, na)).sum()).item() / N          # adds true: new perfect vs old missing na
L_add = s.select((f05(k_, 0, 0) - f05(k_, na, 0)).sum()).item() / N               # adds false: old perfect vs new with na FPs
G_rem = s.select((f05(k_, 0, 0) - f05(k_, nr, 0)).sum()).item() / N               # removed were false: new perfect vs old with FPs
L_rem = s.select((f05(k_ + nr, 0, 0) - f05(k_, 0, nr)).sum()).item() / N         # removed were true: old perfect vs new missing
print(f"France S1 {N}; changed S1 {s.height}; added {add.height}, removed {rem.height}")
print(f"per-French-S1 F0.5: G_add {G_add:.5f} L_add {L_add:.5f} | G_rem {G_rem:.5f} L_rem {L_rem:.5f}")
for pa, pr in [(0.95, 0.95), (0.85, 0.9), (0.75, 0.8), (0.72, 0.72)]:
    d = pa * G_add - (1 - pa) * L_add + pr * G_rem - (1 - pr) * L_rem
    print(f"  precision adds {pa:.2f}, removed-false {pr:.2f}: dF_France {d:+.5f} -> dLB {0.1497 * d:+.6f}")
