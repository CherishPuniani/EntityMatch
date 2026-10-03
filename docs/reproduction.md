# Reproduction guide

## What this snapshot includes

The source, original dependency pins, historical experiments, and two local output TSVs are present. The dataset, official validator, `work/` artifacts, trained models, run logs, and original research reports are absent. The output TSVs are ignored by Git, so someone cloning the repository will receive their manifest and documentation, not the predictions.

## Inputs and working directory

Launch scripts resolve data and generated paths relative to the current directory. Run them from the repository root, with this layout:

```text
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
utils/
  validate_submission.py       Official challenge utility, supplied separately
```

Source TSVs need the columns `entity_id`, `business_name`, `business_address`, and `country`; see `prep.py` for the expected column order. Ground truth uses `source1_entity_id` and comma-separated `matched_entity_ids`. Empty sets are represented by an empty second field. IDs follow `S1-<integer>`, `S2-<integer>`, and `S3-<integer>`.

## Environment

The original implementation notes describe Python 3.13 on Linux, 32 CPU cores, 160 GB RAM, and an RTX 6000 Ada with 48 GB for neural work. Those notes estimate about 3.5 hours for the CPU pipeline plus several GPU hours for extensions. These are historical estimates, not measurements from this cleanup.

`code/business_entity_resolution/requirements.txt` preserves the supplied pins. Package availability, installation, and CUDA compatibility have not been verified. The cleanup used local Python 3.9.6 only for dependency-free checks; it did not install or execute the ML stack. The requirements include both tree and neural dependencies. Install an appropriate CUDA build of PyTorch for GPU runs; the original file mentions the CUDA 12.8 wheel index.

## CPU baseline

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/business_entity_resolution/requirements.txt
NW=2 bash code/business_entity_resolution/src/run_all.sh
```

The launcher copies source modules into `work/`, builds candidates and features, trains both stages, evaluates OOF predictions, writes `output/`, and calls the official validator. `PY` overrides the interpreter; `NW` sets feature-worker concurrency; `PREP_PROCS` controls baseline preprocessing workers. Many other CPU/thread settings are fixed in the source. Reducing `NW` alone does not bound all memory use.

The launcher now checks that inputs and the validator exist before creating a workspace. Feature-worker failures are propagated before training starts. The model computation and settings were preserved; this does not constitute an end-to-end reproduction test.

## Neural extension

After a completed CPU baseline:

```bash
bash code/business_entity_resolution/src/run_neural.sh
```

This requires CUDA and access to the model hub (or already cached models). It produces the `dense` tree system and `dense_ce_qwen` CE system, then country-routed outputs. Step completion markers live in `logs/*.done`. They do not hash configuration or inputs: reuse them only for the same run configuration, and use an isolated workspace for a new run. The scripts overwrite output files; preserve any incumbent predictions before running them.

## Final shift extension: unresolved artifact lineage

`run_shift.sh` describes the intended later workflow, but it expects an intermediate lineage absent from the provided neural launcher:

| Expected by shift extension | Produced by supplied neural launcher |
|---|---|
| `work/oof_dense_s1.parquet` | `work/oof_dense.parquet` |
| `work/models_dense_s1/` | `work/models_dense/` |
| `work/test_scores_dense_s1.parquet` | `work/test_scores_dense.parquet` |
| `work/oof_dense_s1_ce_qwen.parquet` | `work/oof_dense_ce_qwen.parquet` |
| `work/ce/gate_train_h{0,1}_dense_s1.parquet` | `work/ce/gate_train_h{0,1}_dense.parquet` |
| `work/ce/ce_{train,test}_qwen_dense_s1.parquet` | `work/ce/ce_{train,test}_qwen_dense.parquet` |

Recover the original training recipe or artifacts for `dense_s1` before using the final extension. Renaming files alone is not evidence that the models or configurations match. The launcher now fails early when essential inputs are missing. `run_unseen_norm.sh` also requires the `dense_s1` lineage and rebuilds its isolated `work_unseen/` and `fx_unseen/` directories.

Original references to `SOLUTION_REPORT.md`, `RESEARCH_LOG_QWEN3_DITTO.md`, and `RESEARCH_LOG_TEST_SHIFT.md` were retained as historical provenance. Those documents are missing; no links to nonexistent files are presented as working documentation.

## Validation and determinism

```bash
python3 scripts/check_repo.py
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

The first command checks source syntax, shell syntax, and relative Markdown links. It does not run training or validate accuracy. The second requires the official utility and test dataset. The local output audit checks structure and match/candidate consistency but cannot verify dataset membership.

Sampling and tree seeds are fixed in the source. That expresses an intent to reproduce results within the same environment; it is not a claim of verified bit-identical output across environments. Neural training is explicitly described as subject to GPU nondeterminism.
