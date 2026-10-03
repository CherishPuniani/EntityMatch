# Empty-query rescue investigation

This folder contains historical analysis of Source 1 records with no predicted links, especially countries without training labels. Scripts inspect candidate pools, addresses, name edits, score bands, and the tradeoff of adding one link to an empty query.

| Files | Role visible in the source |
|---|---|
| `cls_lib.py`, `cls.py` | Address/name-equality classification helpers and analysis |
| `accd.py`, `moment.py`, `mech.py` | Acceptance, distribution summaries, and mechanism probes |
| `empt.py`, `emp2.py`, `tempty.py` | Empty-query exploration |
| `pool.py`, `samp.py`, `ext.py` | Candidate-pool inspection, sampling, and record extraction |
| `rescue_sel.py`, `rescue_sel_p05.py`, `build9.py` | Alternative rescue selections and output construction |

These scripts contain absolute paths to the original server and temporary scratchpad, import historical helpers, and require intermediate files that are not shipped. The reusable implementation promoted into the main workflow is `src/rescue_empty_unseen.py`, invoked by `src/run_shift.sh` after decoding and duplicate consistency.

No score improvement or final-output lineage can be verified from these files alone. See [the reproduction guide](../../../../../docs/reproduction.md) before attempting a rerun.
