#!/usr/bin/env python3
"""Check source syntax and local Markdown links without importing the ML stack."""
import ast
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "business_entity_resolution"


def main():
    errors = []
    python_files = sorted(SOURCE.rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    shell_files = sorted(SOURCE.rglob("*.sh"))
    markdown_files = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md")) + sorted(SOURCE.rglob("*.md"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    for path in shell_files:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            errors.append(f"{path.relative_to(ROOT)}: {result.stderr.strip()}")
    for path in markdown_files:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local = unquote(target.split("#", 1)[0].strip("<>"))
            if local and not (path.parent / local).exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link target {target}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {len(python_files)} Python files, {len(shell_files)} shell scripts, {len(markdown_files)} Markdown files")
    print("Static checks only; training, dependency installation, and official submission validation were not run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
