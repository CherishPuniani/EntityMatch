# Business Entity Resolution — reproducible pipeline

> **Snapshot status (2026-10-04):** The notes below are preserved from the original implementation. Start with the
> [repository overview](../../README.md), [code map](../../docs/code-map.md), and
> [reproduction guide](../../docs/reproduction.md) for the current inventory and limitations.
> Data, validator, checkpoints, and referenced solution/research logs are missing. In particular,
> `run_neural.sh` writes `dense` artifacts while `run_shift.sh` expects `dense_s1`; the three-command workflow
> below is not currently sufficient to reproduce the final submission. Scores, hardware, timing, dependency
> pins, model licenses, and determinism statements below are original claims, not revalidated results.
> `load.py` prints row counts; it does not assert expected counts. The top-30 candidate description applies
> to the CPU baseline; the neural extension unions additional dense candidates.

Pipeline that regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data only
(no external data). **The submitted files come from three scripts run in order:**

```bash
bash code/business_entity_resolution/src/run_all.sh      # tree pipeline (deterministic, CPU, no network)
bash code/business_entity_resolution/src/run_neural.sh   # Qwen3 dense channel + cross-encoder (1 CUDA GPU)
bash code/business_entity_resolution/src/run_shift.sh    # 4-CE stage 2, test-shift correction, French fixes -> output/
```

The first stage alone (sections below) uses no pretrained language models; the extensions download open models
(Apache-2.0 / MIT, ≤ 0.6B parameters) from the Hugging Face hub.

## Environment

* Python 3.13 (tested), Linux.
* `pip install -r requirements.txt` (pinned: polars, pyarrow, numpy, lightgbm, rapidfuzz, numba, scikit-learn, scipy).
* Reference hardware: 32 CPU cores, 160 GB RAM, 1 × RTX 6000 Ada 48 GB for the extensions. Wall-clock ≈ 3.5 h for
  `run_all.sh`; the GPU extensions add several GPU hours (e.g. in-band Qwen3 cross-encoder training ≈ 1.6 h per half,
  scoring the gated test pairs ≈ 90 min per model).
  Peak memory is dominated by the parallel feature workers (~26 GB each, `NW=5` by default) and
  stage training (~80 GB). With less RAM, lower `NW` (e.g. `NW=2`); results are identical.

## Run

From the challenge root (the directory that contains `dataset/` and `utils/`):

```bash
bash code/business_entity_resolution/src/run_all.sh
```

Intermediate artefacts are written to `work/`, final files to `output/`. The script ends by running
`utils/validate_submission.py --check-ids`.

## Steps (src/)

| Step | Script | What it does |
|---|---|---|
| 1 | `load.py` | TSV → parquet (tab separator, quoting disabled, row counts verified) |
| 2 | `learn_dict.py` | learns the native-script (Devanagari, Telugu, Tamil, …) → Latin word dictionary from aligned **training** ground-truth pairs |
| 3 | `prep.py` + `norm.py` | normalised name/address forms: Unicode folding, transliteration, de-leetspeak, legal/honorific/filler stripping, alias splitting, address abbreviation canonicalisation, state codes, PO-box removal |
| 4 | `gt_pairs.py` | ground-truth pair table |
| 5 | `raw_pool.py`, `train_ranker.py` | full raw blocking pool for 5 % of train S1 (99.4 % recall) → LightGBM **blocking ranker** over per-key-type IDF sums |
| 6 | `run_block.py` + `block.py` | compound hashed blocking keys, IDF weights, learned ranking → top-30 candidates per S1; target-side competition features. For training, a deterministic 18.6 % of train S1 is removed (`keep_trainD.parquet`) to reproduce the test-set S1/target ratio |
| 7 | `rec_attrs.py` | per-record hashes (core name, address, house number) and name/address frequencies |
| 8 | `build_feats_all.py` + `features.py` | ~115 pair features (string similarity, IDF token sets, house-number relations, blocking and competition features); chunked, multi-process |
| 9 | `train_stages.py` + `stages.py` | two-stage LightGBM with 4 S1-grouped folds; stage 2 adds probability-based context (target competition, within-S1 rank, cluster corroboration) |
| 10 | `predict_test.py` | averages the 4 fold models per stage, applies threshold 0.7 + target exclusivity, writes both TSVs |
| – | `eval_oof.py`, `metric.py`, `policy.py` | exact macro-F0.5 evaluation and decision-policy experiments |

`candidate_pairs.tsv` contains exactly the (S1, target) pairs scored by the stage-1 model
(top-30 per S1 after learned blocking); every match in `matching_results.tsv` is a subset of it.

## Determinism

All sampling is hash-based with fixed seeds, and LightGBM uses fixed seeds. Polars joins that feed model
matrices use `maintain_order="left"`.

## Neural extension (Qwen3 cross-encoder + Qwen3-Embedding candidate channel)

Added 2026-09-26; see `../SOLUTION_REPORT.md` §28 and `../RESEARCH_LOG_QWEN3_DITTO.md`. It runs **after**
`run_all.sh` and needs one CUDA GPU (reference: RTX 6000 Ada 48 GB; peak 29 GB) plus the pinned torch /
transformers packages in `requirements.txt`. Models are pulled from the Hugging Face hub (Apache-2.0):
`Qwen/Qwen3-Reranker-0.6B` (595.8M parameters as deployed) and `Qwen/Qwen3-Embedding-0.6B`.

```bash
bash code/business_entity_resolution/src/run_all.sh      # tree pipeline (TAG=final)
bash code/business_entity_resolution/src/run_neural.sh   # neural extension -> output/
```

| Step | Script | What it does |
|---|---|---|
| N1 | `nn_ce_data.py` | cross-encoder training pairs from the blocking candidates: per half 40k S1, positives + top-6 retrieved non-matches + 1 lower-ranked |
| N2 | `nn_ce.py train` | Ditto-style serialized pairs → Qwen3-Reranker-0.6B with a binary head (`w_yes − w_no`), BCE, 1 epoch; one model per half |
| N3 | `nn_ce_gate.py` + `nn_ce.py score` | gate 0.005 ≤ stage-1 p1 ≤ 0.995; train pairs scored by the other half's model (out-of-fold), test by the half-0 model |
| N4 | `train_stage2_ce.py` | stage 2 re-trained on the unchanged stage-1 OOF p1 with CE features (`CE_WITHIN=1`: within-S1 features only) |
| N5 | `dense_cands.py`, `nn_dense.py` | Qwen3-Embedding-0.6B top-10 per S1 over targets not assigned by the base model → extra candidates (`NOFILTER=1 build_feats_all.py`) |
| N6 | `train_stages.py`, `predict_test.py` | stages 1+2 re-trained on blocking ∪ dense candidates (`FEAT` accepts several dirs, `EXTRA_PAIRFEATS` adds dscore/drank/dense_only) |
| N7 | `nn_ce_regate.py` | re-gate on the new p1; only newly gated pairs are scored |
| N8 | `route_by_country.py` | S1s of countries present in the training labels use the neural system; other countries use the tree-only p2 |
| – | `compare_oof.py`, `analysis_recall.py`, `analysis_dense.py`, `test_readout.py`, `p1_by_parts.py` | paired-bootstrap evaluation, recall diagnostics, label-free test readouts, streamed stage-1 scoring |

Determinism: CE training uses fixed torch/numpy seeds, but cuDNN/SDPA kernels are not bit-deterministic; re-training
moves CE scores slightly. The shipped outputs come from the run logged in `RESEARCH_LOG_QWEN3_DITTO.md`.

## Test-shift extension (second 2026-09-26 session)

See `../RESEARCH_LOG_TEST_SHIFT.md`. It runs **after** `run_all.sh` and `run_neural.sh` and writes `output/`: US/India scored by the 4-CE stage 2, countries without training labels by the tree on
re-normalised features, then the label-free correction and veto (see `../SOLUTION_REPORT.md` §30).

```bash
bash code/business_entity_resolution/src/run_shift.sh
```

| Step | Script | What it does |
|---|---|---|
| A | `nn_ce.py score`, `ce_avg.py` | second Qwen3 half-model on the test gate; test CE = mean of both halves |
| B | `nn_ce.py train/score` | XLM-R-base cross-encoder (E4 recipe) scored on the final gate (OOF on train, both halves on test) |
| C | `nn_ce.py train/score` | **in-band** cross-encoders (XLM-R-base and Qwen3-Reranker-0.6B): trained on the labelled gated band (0.005 ≤ p1 ≤ 0.995) of their half, cross-fitted; test = mean of both halves |
| D | `train_stage2_ce2.py`, `predict_ce2.py` | stage 2 with four CE blocks (Qwen3, XLM-R, in-band XLM-R, in-band Qwen3; within-S1 features), SUB 1.0 — OOF 0.99023 |
| E0 | `run_unseen_norm.sh` | countries without training labels (read from the data): candidate pairs re-featurised with an extended normaliser (dotted legal forms, `bd`→`blvd`, French regions/departments as state codes; French fillers/legal words mapped onto their training-vocabulary equivalents, centre/service out of the filler list) and scored by the tree `dense_s1` |
| E1 | `cells.py` | per cell (country × house-number relation × p2 bin): train positives per S1 vs test pairs per S1 |
| E2 | `apply_shift.py` | label-shift correction of adjacent-address sibling strata (training countries: own rates, p2 0.5–0.95, ≥ 2× excess; unlabelled countries: pooled rates × copy-typo scale, gated by a legal-form mixture test) |
| E3 | `apply_lfveto.py`, `lfutil.py` | legal-form-change veto for unlabelled countries that pass the same label-free test |
| F1 | `apply_twveto.py` | type-word-swap veto for unlabelled countries (same address; type-word vocabulary learnt label-free from replace:append ≥ 10) |
| F2 | `efdec.py`, `final_ef.py` | expected-F0.5 decoding per S1 for countries with training labels (OOF +0.00006) |
| F3 | `dupfix.py` | exact-duplicate target records linked to the same S1 (65 test links) |
| G | `rescue_empty_unseen.py` | unlabelled countries: S1s without links get their unclaimed same-address best candidate when the name edit is filler-only / legal form added / identical (513 test links; writes `output/`) |
| – | `diag_expf.py`, `diag_unc.py`, `diag_cluster.py`, `negctrl.py`, `sibvocab.py`, `mask_country.py` | diagnostics, negative control, rejected sibling-vocabulary veto, leaderboard country probe |

No test labels are used or inferred. Test records only enter as unlabeled inputs, as the IDF and competition
features already did, and every country-specific decision is read from the data.
