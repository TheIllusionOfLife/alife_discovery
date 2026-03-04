#!/usr/bin/env python3
"""Fail when key documentation files reference non-existent local paths."""

from __future__ import annotations

import re
from pathlib import Path

DOC_FILES = (Path("README.md"), Path("STRUCTURE.md"), Path("docs/repro_paper_anonymous.md"))
PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+)`")
CHECK_EXTS = {".md", ".py", ".json", ".tex", ".bib", ".toml", ".yml", ".yaml"}
ROOT_PREFIXES = (
    "alife_discovery/",
    "docs/",
    "scripts/",
    "paper/",
    "tests/",
    ".github/",
)


def _should_check(token: str) -> bool:
    if token.startswith(("http://", "https://")):
        return False
    if token.startswith(("data/", "output/", "legacy/", "docs/legacy/")):
        return False
    path = Path(token)
    if "/" not in token:
        return False
    if not token.startswith(ROOT_PREFIXES):
        return False
    if path.suffix not in CHECK_EXTS:
        return False
    if "<" in token or ">" in token or "*" in token:
        return False
    return True


def main() -> int:
    repo_root = Path.cwd()
    missing: list[tuple[Path, str]] = []

    for doc in DOC_FILES:
        if not doc.exists():
            missing.append((doc, str(doc)))
            continue
        for token in PATH_RE.findall(doc.read_text()):
            if not _should_check(token):
                continue
            resolved = (repo_root / token).resolve()
            if not resolved.exists():
                missing.append((doc, token))

    if missing:
        for doc, token in missing:
            print(f"missing path in {doc}: {token}")
        return 1

    print("doc path check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
