# France-only re-featurisation (iteration 2, experimental)

> Historical scripts; inputs and referenced research logs are absent from this snapshot.
> See the [current reproduction guide](../../../../docs/reproduction.md). The replacement normalizer
> still needs the later `dense_s1` model lineage and is not independently runnable from a fresh clone.

**Superseded by `../run_unseen_norm.sh`** (same fixes, relative paths, unseen countries read from the data; it reproduces
these scripts' French scores exactly). Kept as the record of the 2026-09-26/27 runs.

Scripts as run on 2026-09-26 (absolute paths of that run; see `RESEARCH_LOG_TEST_SHIFT.md` §11–§13).
`run_frnorm2.sh` builds an isolated `run/work4` (symlinks to read-only inputs) with a patched `norm.py`:
dotted legal forms collapsed (`s.a.r.l.` → `sarl`), `bd`/`boul`/`bld` → `blvd`, French regions and departments
mapped to region codes that are dropped from the core address like US/Indian state codes. It re-prepares French
records only, recomputes `rec_attrs`, then `redo_frnorm2.sh` recomputes features for the **original** French candidate
pairs (reconstructed from the original `feat_test*` parts — do not use the regenerated `work/pairs_test.parquet`) and
re-scores them with the tree model `dense_s1`; `post_frnorm2.sh` merges, recomputes French cells, applies the
correction + legal-form veto and validates (`output_shift_frnorm2/`). `build_frens.sh` builds the (not recommended)
France tree + 3-CE average variant. Not part of `run_shift.sh`: to adopt the fixes for good, move them into
`norm.py` and re-featurise France (US/India features would change slightly for dotted forms such as `P.L.L.C.`).
