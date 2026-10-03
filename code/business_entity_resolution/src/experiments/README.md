# Experiment guide

These scripts preserve the exploratory work around the main matching system. They are not a batch of required production stages. Many read historical run directories or scratchpad modules that are absent from this snapshot. They are useful for understanding the questions investigated; reproduce them only after reviewing their inputs and paths.

The original research logs referenced in comments are missing. Descriptions of accepted or rejected variants below come from the available source comments and existing notes, not a rerun of the experiments.

| Group | Files | Investigation |
|---|---|---|
| Feature comparison | `featcmp.py`, `dump_s1.py` | Inspect feature matrices and stage-1 scores |
| Country comparison | `cmp_france.py`, `fr_diag.py`, `diag_expf2.py` | French decision behavior and model-implied expected metric |
| Retrieval probes | `fr_recall_rule.py`, `token_cluster.py`, `dropadd.py` | Same-address recall, token clusters, and additions/removals |
| Model variants | `train_stages_nob.py`, `resume_nob.py`, `predict_nob.py`, `train_stage2_ce2b.py` | Alternate training/features and additional cross-encoder blocks |
| French iterations | [fr_iter4](fr_iter4/README.md) | Normalization, copy/sibling signatures, vetoes, decoding, and retrieval ideas |
| Empty-query rescue | [iter6_rescue](iter6_rescue/README.md) | Candidate and name-edit analysis for queries with no predicted matches |

Related work outside this directory:

- [Historical French normalization](../france_norm/README.md) is superseded by `src/run_unseen_norm.sh`.
- `analysis_recall.py`, `analysis_dense.py`, and `analysis_dense_test.py` measure retrieval coverage and dense-channel behavior.
- `compare_oof.py` evaluates paired systems on a fixed population with bootstrap comparisons.
- `diag_unc.py`, `diag_expf.py`, `diag_cluster.py`, and `negctrl.py` investigate uncertainty and test-shift assumptions.
- `sibvocab.py` implements a sibling-vocabulary veto described as rejected in its comments.
- `mask_country.py` creates a country-masked submission for a historical leaderboard probe. It is a diagnostic, not part of the final inference path.

Use the [code map](../../../../docs/code-map.md) for the main workflow and the [reproduction guide](../../../../docs/reproduction.md) for current limitations.
