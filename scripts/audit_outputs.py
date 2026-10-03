#!/usr/bin/env python3
"""Stream local TSV outputs and compare their structure and hashes to the manifest.

This is a snapshot-integrity check, not the official challenge validator.
Run from any directory: python3 scripts/audit_outputs.py
"""
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def audit():
    names = ("matching_results.tsv", "candidate_pairs.tsv")
    expected = json.loads((ROOT / "docs/output_manifest.json").read_text())["files"]
    paths = [ROOT / "output" / name for name in names]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Local output files are absent: " + ", ".join(missing))
    hashers = [hashlib.sha256(), hashlib.sha256()]
    stats = [dict(rows=0, nonempty_rows=0, pairs=0, max_targets_per_row=0) for _ in names]
    headers = (b"source1_entity_id\tmatched_entity_ids\n", b"source1_entity_id\tcandidate_entity_ids\n")
    seen = set()
    with paths[0].open("rb") as matching, paths[1].open("rb") as candidates:
        for index, file in enumerate((matching, candidates)):
            header = file.readline()
            hashers[index].update(header)
            if header != headers[index]:
                raise ValueError("Unexpected header in " + names[index])
        for row, lines in enumerate(itertools.zip_longest(matching, candidates), start=2):
            parsed = []
            for index, line in enumerate(lines):
                if line is None:
                    raise ValueError(f"Row count mismatch at row {row}")
                hashers[index].update(line)
                columns = line.decode("utf-8").rstrip("\r\n").split("\t")
                if len(columns) != 2:
                    raise ValueError(f"Expected two columns at row {row} in {names[index]}")
                source, values = columns
                targets = values.split(",") if values else []
                if not re.fullmatch(r"S1-\d+", source) or any(not re.fullmatch(r"S[23]-\d+", target) for target in targets):
                    raise ValueError(f"Invalid ID format at row {row}")
                if len(set(targets)) != len(targets):
                    raise ValueError(f"Duplicate target within row {row}")
                summary = stats[index]
                summary["rows"] += 1
                summary["nonempty_rows"] += bool(targets)
                summary["pairs"] += len(targets)
                summary["max_targets_per_row"] = max(summary["max_targets_per_row"], len(targets))
                parsed.append((source, set(targets)))
            if parsed[0][0] != parsed[1][0]:
                raise ValueError(f"Source order mismatch at row {row}")
            if parsed[0][0] in seen:
                raise ValueError(f"Duplicate source at row {row}")
            seen.add(parsed[0][0])
            if not parsed[0][1] <= parsed[1][1]:
                raise ValueError(f"Selected match absent from candidates at row {row}")
    for index, name in enumerate(names):
        actual = dict(bytes=paths[index].stat().st_size, sha256=hashers[index].hexdigest(), **stats[index])
        if actual != expected[name]:
            raise ValueError("Output differs from documented snapshot: " + name)
    print("PASS: TSV structure, source alignment, match/candidate containment, and manifest hashes")
    print("Dataset ID membership and accuracy were not checked.")


if __name__ == "__main__":
    try:
        audit()
    except (ValueError, OSError, UnicodeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
