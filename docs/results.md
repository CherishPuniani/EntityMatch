# Results and evidence

## Accuracy claims

The preserved implementation README reports **0.99023 OOF macro F0.5** for the four-cross-encoder stage-2 system and an additional **0.00006 OOF** from expected-F0.5 decoding. These are historical claims. This snapshot contains neither the supporting OOF predictions nor training/evaluation logs, so the scores were not reproduced during cleanup.

Comments in `run_unseen_norm.sh` report a leaderboard change from 0.984851 to 0.98667 during French normalization work. No leaderboard export, original research log, or verified final competition score is included. This should not be presented as a verified final score.

The original notes also describe 99.4% recall for a sampled raw blocking pool. That is not a verified final test-candidate recall figure. Match-set correctness and final test accuracy cannot be inferred from the output structure.

## Local output audit

On 2026-10-04, both local output files were streamed in full. The audit checked exact headers, ID format, unique Source 1 IDs, no repeated target IDs within a row, identical Source 1 IDs and row order across files, and that every selected match appears in the corresponding candidate list. All those checks passed.

| Property | Matching results | Candidate lists |
|---|---:|---:|
| Source 1 rows | 1,732,544 | 1,732,544 |
| Rows with at least one target | 1,631,275 | 1,732,542 |
| Total query–target pairs | 5,813,732 | 65,163,241 |
| Maximum targets in one row | 11 | 41 |
| File bytes | 97,365,085 | 862,216,745 |

The candidate maximum of 41 confirms that the local files are not simply a fixed top-30 baseline output. It does not identify the exact model run that produced them. Producing-run provenance cannot be recovered from hashes alone.

The [manifest](output_manifest.json) contains SHA-256 hashes and the audit summary. Large predictions stay in local `output/` and are ignored by Git. No real business records or ID excerpts are added to the documentation.

To repeat the structure and snapshot-integrity audit when the local files are available:

```bash
python3 scripts/audit_outputs.py
```

This streams both files and retains the Source 1 IDs for uniqueness checks. It does not load all candidate pairs into memory.

## File formats

Both files are TSVs with one Source 1 entity per row. The second field is a comma-separated target-ID list, empty when no target is selected. These examples use invented IDs:

```text
source1_entity_id	matched_entity_ids
S1-100	S2-200,S3-300
S1-101	
```

```text
source1_entity_id	candidate_entity_ids
S1-100	S2-200,S3-300,S2-201
S1-101	S3-301
```

## Cleanup verification and limits

- All 122 original Python files parse; all 12 shell scripts pass `bash -n`.
- The added repository checker validates source syntax and local Markdown links without importing the ML stack.
- Launcher preflight checks reject absent inputs before creating generated workspaces.
- The CPU launcher waits for individual feature workers and propagates their failures before training.
- A targeted text scan found no common credential patterns in the supplied code. This is not an exhaustive secret audit.

The dataset and official validator are missing, so ID membership and official challenge compliance were not checked. No dependencies were installed, no model was trained, and no CPU/GPU end-to-end run was performed. The final `dense` versus `dense_s1` artifact mismatch remains documented in the [reproduction guide](reproduction.md).
