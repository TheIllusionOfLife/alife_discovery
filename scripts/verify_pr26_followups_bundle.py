"""Verify consistency of PR26 follow-up manifest/checksums bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.pr26_followups_manifest_paths import collect_manifest_output_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checksums(checksums_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in checksums_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid checksum line: {line!r}")
        entries[parts[1].strip()] = parts[0].strip()
    return entries


def _resolve_bundle_path(
    *,
    followup_dir: Path,
    rel_path: str,
    errors: list[str],
    context: str,
) -> Path | None:
    candidate = (followup_dir / rel_path).resolve()
    try:
        candidate.relative_to(followup_dir)
    except ValueError:
        errors.append(f"{context} escapes bundle directory: {rel_path}")
        return None
    return candidate


def verify_bundle(followup_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    followup_dir = followup_dir.resolve()
    manifest_path = followup_dir / "manifest.json"
    checksums_path = followup_dir / "checksums.sha256"
    if not manifest_path.exists():
        return False, [f"Missing manifest: {manifest_path}"]
    if not checksums_path.exists():
        return False, [f"Missing checksums: {checksums_path}"]

    manifest = json.loads(manifest_path.read_text())
    output_paths, skipped = collect_manifest_output_paths(
        manifest,
        manifest_path,
        base_dir=followup_dir,
    )
    if any(skipped.values()):
        errors.append(f"Manifest outputs had skipped entries: {skipped}")

    expected_rel_paths = {"manifest.json"}
    for path in output_paths:
        expected_rel_paths.add(str(path.resolve().relative_to(followup_dir)))

    checksum_entries = _parse_checksums(checksums_path)
    checksum_paths = set(checksum_entries.keys())
    if checksum_paths != expected_rel_paths:
        missing = sorted(expected_rel_paths - checksum_paths)
        extra = sorted(checksum_paths - expected_rel_paths)
        if missing:
            errors.append(f"checksums missing expected entries: {missing}")
        if extra:
            errors.append(f"checksums has unexpected entries: {extra}")

    for rel_path, expected_hash in checksum_entries.items():
        target = _resolve_bundle_path(
            followup_dir=followup_dir,
            rel_path=rel_path,
            errors=errors,
            context="checksums entry",
        )
        if target is None:
            continue
        if not target.exists():
            errors.append(f"checksums entry points to missing file: {rel_path}")
            continue
        actual_hash = _sha256(target)
        if actual_hash != expected_hash:
            errors.append(
                f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
            )

    zenodo = manifest.get("zenodo")
    if isinstance(zenodo, dict):
        files = zenodo.get("files", [])
        if isinstance(files, list):
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    continue
                rel_path = file_entry.get("relative_path")
                if not isinstance(rel_path, str):
                    continue
                local_path = _resolve_bundle_path(
                    followup_dir=followup_dir,
                    rel_path=rel_path,
                    errors=errors,
                    context="zenodo relative_path",
                )
                if local_path is None:
                    continue
                if not local_path.exists():
                    continue
                recorded_hash = file_entry.get("sha256")
                # checksums.sha256 intentionally omits recorded hash to avoid circular coupling.
                if rel_path == "checksums.sha256":
                    continue
                if recorded_hash is None:
                    errors.append(f"zenodo.files missing sha256 for {rel_path}")
                    continue
                if not isinstance(recorded_hash, str):
                    errors.append(f"zenodo.files sha256 is not string for {rel_path}")
                    continue
                actual_hash = _sha256(local_path)
                if actual_hash != recorded_hash:
                    errors.append(
                        "Zenodo hash mismatch for "
                        f"{rel_path}: expected {recorded_hash}, got {actual_hash}"
                    )

    return len(errors) == 0, errors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify PR26 follow-up bundle integrity")
    parser.add_argument("--followup-dir", type=Path, default=Path("data/post_hoc/pr26_followups"))
    args = parser.parse_args(argv)

    ok, errors = verify_bundle(args.followup_dir)
    if ok:
        print("Bundle verification: OK")
        return
    print("Bundle verification: FAILED")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
