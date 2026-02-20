"""Run all PR #26 follow-up analyses and emit a single manifest.

Usage:
    uv run python scripts/run_pr26_followups.py \
      --data-dir data/stage_d \
      --out-dir data/post_hoc/pr26_followups
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.no_filter_analysis import main as run_no_filter
from scripts.phenotype_taxonomy import main as run_taxonomy
from scripts.ranking_stability import main as run_ranking_stability
from scripts.synchronous_ablation import main as run_sync_ablation
from scripts.te_null_analysis import main as run_te_null


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run all PR #26 follow-up analyses")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage_d"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/pr26_followups"))
    parser.add_argument("--quick", action="store_true", help="Run quick presets for all analyses")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    checksums_path = out_dir / "checksums.sha256"

    def _run_output(command: list[str]) -> str:
        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    branch_name = _run_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    command_line = " ".join(
        [
            "uv",
            "run",
            "python",
            "scripts/run_pr26_followups.py",
            *[shlex.quote(arg) for arg in (argv or [])],
        ]
    )

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "quick": args.quick,
        "git_commit": "unknown",
        "git_branch": branch_name,
        "command_line": command_line,
        "python_version": sys.version.split(" ", maxsplit=1)[0],
        "uv_version": _run_output(["uv", "--version"]),
        "platform": platform.platform(),
        "commands": {},
        "outputs": {},
        "analysis_status": {},
        "zenodo": None,
    }

    def _write_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_checksums() -> None:
        out_dir_resolved = out_dir.resolve()
        targets: list[Path] = [manifest_path]

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
        if isinstance(outputs, dict):
            for entry in outputs.values():
                if not isinstance(entry, dict):
                    continue
                for key in ("json", "csv"):
                    path_text = entry.get(key)
                    if not isinstance(path_text, str):
                        continue
                    resolved = _resolve_output_path(path_text)
                    if resolved is None:
                        continue
                    if not resolved.is_file():
                        continue
                    try:
                        resolved.relative_to(out_dir_resolved)
                    except ValueError:
                        continue
                    if resolved not in targets:
                        targets.append(resolved)
        lines = [
            f"{_sha256(path)}  {path.resolve().relative_to(out_dir_resolved)}" for path in targets
        ]
        checksums_path.write_text("\n".join(lines) + "\n")

    no_filter_dir = out_dir / "no_filter"
    no_filter_args = ["--out-dir", str(no_filter_dir)]
    if args.quick:
        no_filter_args.append("--quick")
    manifest["commands"]["no_filter"] = [
        "uv",
        "run",
        "python",
        "scripts/no_filter_analysis.py",
        *no_filter_args,
    ]
    try:
        run_no_filter(no_filter_args)
        manifest["analysis_status"] = {**manifest["analysis_status"], "no_filter": "success"}
    except Exception as exc:
        manifest["analysis_status"] = {
            **manifest["analysis_status"],
            "no_filter": f"failed: {exc}",
        }
    manifest["outputs"] = {
        **manifest["outputs"],
        "no_filter": {
            "json": str(no_filter_dir / "summary.json"),
            "csv": str(no_filter_dir / "summary.csv"),
        },
    }
    _write_manifest()

    sync_dir = out_dir / "synchronous_ablation"
    sync_args = ["--out-dir", str(sync_dir)]
    if args.quick:
        sync_args.append("--quick")
    manifest["commands"] = {
        **manifest["commands"],
        "synchronous_ablation": [
            "uv",
            "run",
            "python",
            "scripts/synchronous_ablation.py",
            *sync_args,
        ],
    }
    try:
        run_sync_ablation(sync_args)
        manifest["analysis_status"] = {
            **manifest["analysis_status"],
            "synchronous_ablation": "success",
        }
    except Exception as exc:
        manifest["analysis_status"] = {
            **manifest["analysis_status"],
            "synchronous_ablation": f"failed: {exc}",
        }
    manifest["outputs"] = {
        **manifest["outputs"],
        "synchronous_ablation": {
            "json": str(sync_dir / "summary.json"),
            "csv": str(sync_dir / "summary.csv"),
        },
    }
    _write_manifest()

    ranking_dir = out_dir / "ranking_stability"
    ranking_args = ["--out-dir", str(ranking_dir)]
    if args.quick:
        ranking_args.append("--quick")
    manifest["commands"] = {
        **manifest["commands"],
        "ranking_stability": [
            "uv",
            "run",
            "python",
            "scripts/ranking_stability.py",
            *ranking_args,
        ],
    }
    try:
        run_ranking_stability(ranking_args)
        manifest["analysis_status"] = {
            **manifest["analysis_status"],
            "ranking_stability": "success",
        }
    except Exception as exc:
        manifest["analysis_status"] = {
            **manifest["analysis_status"],
            "ranking_stability": f"failed: {exc}",
        }
    manifest["outputs"] = {
        **manifest["outputs"],
        "ranking_stability": {
            "json": str(ranking_dir / "summary.json"),
            "csv": str(ranking_dir / "summary.csv"),
        },
    }
    _write_manifest()

    te_dir = out_dir / "te_null"
    te_args = [
        "--data-dir",
        str(args.data_dir),
        "--out-dir",
        str(te_dir),
    ]
    if args.quick:
        te_args.append("--quick")
    manifest["commands"] = {
        **manifest["commands"],
        "te_null": ["uv", "run", "python", "scripts/te_null_analysis.py", *te_args],
    }
    try:
        run_te_null(te_args)
        manifest["analysis_status"] = {**manifest["analysis_status"], "te_null": "success"}
    except Exception as exc:
        manifest["analysis_status"] = {**manifest["analysis_status"], "te_null": f"failed: {exc}"}
    manifest["outputs"] = {
        **manifest["outputs"],
        "te_null": {
            "json": str(te_dir / "summary.json"),
            "csv": str(te_dir / "summary.csv"),
        },
    }
    _write_manifest()

    taxonomy_dir = out_dir / "phenotypes"
    taxonomy_args = [
        "--data-dir",
        str(args.data_dir),
        "--out-dir",
        str(taxonomy_dir),
    ]
    if args.quick:
        taxonomy_args.append("--quick")
    manifest["commands"] = {
        **manifest["commands"],
        "phenotype_taxonomy": [
            "uv",
            "run",
            "python",
            "scripts/phenotype_taxonomy.py",
            *taxonomy_args,
        ],
    }
    try:
        run_taxonomy(taxonomy_args)
        manifest["analysis_status"] = {
            **manifest["analysis_status"],
            "phenotype_taxonomy": "success",
        }
    except Exception as exc:
        manifest["analysis_status"] = {
            **manifest["analysis_status"],
            "phenotype_taxonomy": f"failed: {exc}",
        }
    manifest["outputs"] = {
        **manifest["outputs"],
        "phenotype_taxonomy": {
            "json": str(taxonomy_dir / "taxonomy.json"),
            "csv": str(taxonomy_dir / "taxonomy.csv"),
        },
    }
    _write_manifest()

    git_commit = _run_output(["git", "rev-parse", "HEAD"])

    manifest["git_commit"] = git_commit
    _write_manifest()
    _write_checksums()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
