# Code map

Paths below are relative to `code/business_entity_resolution/src/`. Read the launchers first to see execution order; read the shared modules to understand implementation. Most entry points are scripts with positional arguments or environment variables, rather than an installable Python package.

## Launchers

| File | Purpose |
|---|---|
| `run_all.sh` | Data preparation → lexical candidates → pair features → tree training → baseline outputs |
| `run_neural.sh` | Gated Qwen3 cross-encoder → dense candidates → retraining → country routing |
| `run_shift.sh` | Later four-CE stacking, shift correction, vetoes, decoding, and rescue; requires missing `dense_s1` lineage |
| `run_unseen_norm.sh` | Isolated re-normalization and tree rescoring for countries absent from training labels |

## Data, normalization, and candidate retrieval

| Files | Responsibility |
|---|---|
| `load.py`, `ids.py`, `gt_pairs.py` | Input conversion, reversible source-aware integer IDs, ground-truth pairs |
| `learn_dict.py`, `norm.py`, `prep.py` | Learned transliteration, text normalization, prepared record views |
| `make_keep.py` | Deterministic Source 1 training dropout for target/query-ratio matching |
| `block.py`, `raw_pool.py`, `train_ranker.py`, `run_block.py` | Blocking keys, ranker training, and top-K lexical retrieval |
| `rec_attrs.py`, `features.py`, `build_feats_all.py` | Record attributes and chunked pair-feature construction |

## Tree models and evaluation

| Files | Responsibility |
|---|---|
| `stages.py`, `train_stages.py`, `predict_test.py` | Shared contextual features, grouped OOF training, averaged test prediction |
| `metric.py`, `policy.py`, `eval_oof.py` | Exact macro F0.5 and candidate-to-match-set policies |
| `compare_oof.py` | Fixed-population paired evaluation and bootstrap comparison |
| `p1_by_parts.py` | Streamed stage-1 inference to reduce peak memory |

## Neural models

| Files | Responsibility |
|---|---|
| `nn_common.py` | Model names, record loading, serialization, and neural ID helpers |
| `nn_ce_data.py`, `nn_ce.py` | Retrieved-negative training data, CE training, and pair scoring |
| `nn_ce_gate.py`, `nn_ce_regate.py` | Probability gates, cross-fitted score assembly, reuse after candidate expansion |
| `nn_dense.py`, `dense_cands.py` | Embedding-based search and lexical/dense candidate union |
| `train_stage2_ce.py`, `predict_test_ce.py` | One-CE stage-2 stacking and inference |
| `train_stage2_ce2.py`, `predict_ce2.py` | Multiple-CE stage-2 stacking and inference, including optional in-band blocks |
| `ce_avg.py`, `xl_assemble.py` | Two-half test-score averaging and XLM-R score assembly for the later lineage |

## Routing, correction, and final decisions

| Files | Responsibility |
|---|---|
| `route_by_country.py` | Select neural or tree scores according to country coverage in training |
| `cells.py`, `apply_shift.py` | Estimate and apply selected distribution-shift adjustments |
| `lfutil.py`, `apply_lfveto.py` | Legal-form parsing and change veto |
| `apply_twveto.py` | Type-word replacement veto for unseen countries |
| `efdec.py`, `final_ef.py` | Expected-F0.5 match-set decoding and final integration |
| `dupfix.py`, `rescue_empty_unseen.py` | Duplicate-target consistency and selected empty-query rescue |

## Diagnostics and research history

| Files | Responsibility |
|---|---|
| `analysis_recall.py`, `analysis_dense.py`, `analysis_dense_test.py` | Candidate recall and dense-channel analysis |
| `test_readout.py`, `diag_unc.py` | Label-free test readouts and uncertain-pair strata |
| `diag_expf.py`, `diag_cluster.py` | Model-implied expected metric and sibling-cluster diagnostics |
| `negctrl.py` | Treat a labeled country as unseen to probe correction assumptions |
| `sibvocab.py` | Historical sibling-vocabulary veto described as rejected |
| `mask_country.py` | Country-masked output for a historical leaderboard diagnostic |
| [experiments/](../code/business_entity_resolution/src/experiments/README.md) | Alternate models, country analysis, and rescue probes |
| [france_norm/](../code/business_entity_resolution/src/france_norm/README.md) | Superseded French normalization launchers |

All top-level Python and shell files are covered above. Experimental subdirectories have their own guides. Their presence records an investigation; it does not imply that every script contributed to the shipped outputs.
