"""Shared manifest output path discovery for PR26 follow-up scripts."""

from __future__ import annotations

from pathlib import Path


def collect_manifest_output_paths(
    manifest: dict,
    manifest_path: Path,
    *,
    base_dir: Path | None = None,
) -> tuple[list[Path], dict[str, int]]:
    """Resolve manifest-declared output JSON/CSV paths.

    Relative paths are resolved in a backward-compatible order:
    1) relative to manifest file directory
    2) relative to current working directory
    """

    skipped = {
        "non_string_path": 0,
        "resolve_error": 0,
        "outside_base_dir": 0,
        "missing_file": 0,
    }
    targets: list[Path] = []
    resolved_base = base_dir.resolve() if base_dir is not None else None

    def _resolve_output_path(path_text: str) -> Path | None:
        candidate = Path(path_text)
        if candidate.is_absolute():
            try:
                return candidate.resolve()
            except OSError:
                return None
        manifest_relative = manifest_path.parent / candidate
        try:
            manifest_resolved = manifest_relative.resolve()
        except OSError:
            manifest_resolved = None
        if manifest_resolved is not None and manifest_resolved.is_file():
            return manifest_resolved
        try:
            return (Path.cwd() / candidate).resolve()
        except OSError:
            return None

    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        return targets, skipped

    for entry in outputs.values():
        if not isinstance(entry, dict):
            continue
        for key in ("json", "csv"):
            path_text = entry.get(key)
            if not isinstance(path_text, str):
                skipped["non_string_path"] += 1
                continue
            resolved = _resolve_output_path(path_text)
            if resolved is None:
                skipped["resolve_error"] += 1
                continue
            if not resolved.is_file():
                skipped["missing_file"] += 1
                continue
            if resolved_base is not None:
                try:
                    resolved.relative_to(resolved_base)
                except ValueError:
                    skipped["outside_base_dir"] += 1
                    continue
            if resolved not in targets:
                targets.append(resolved)

    return targets, skipped
