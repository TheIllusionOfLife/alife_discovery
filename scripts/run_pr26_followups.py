"""Run all PR #26 follow-up analyses and emit a single manifest.

Usage:
    uv run python scripts/run_pr26_followups.py \
      --data-dir data/stage_d \
      --out-dir data/post_hoc/pr26_followups
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

    no_filter_dir = out_dir / "no_filter"
    no_filter_args = ["--out-dir", str(no_filter_dir)]
    if args.quick:
        no_filter_args.append("--quick")
    run_no_filter(no_filter_args)

    sync_dir = out_dir / "synchronous_ablation"
    sync_args = ["--out-dir", str(sync_dir)]
    if args.quick:
        sync_args.append("--quick")
    run_sync_ablation(sync_args)

    ranking_dir = out_dir / "ranking_stability"
    ranking_args = ["--out-dir", str(ranking_dir)]
    if args.quick:
        ranking_args.append("--quick")
    run_ranking_stability(ranking_args)

    te_dir = out_dir / "te_null"
    te_args = [
        "--data-dir",
        str(args.data_dir),
        "--out-dir",
        str(te_dir),
    ]
    if args.quick:
        te_args.append("--quick")
    run_te_null(te_args)

    taxonomy_dir = out_dir / "phenotypes"
    taxonomy_args = [
        "--data-dir",
        str(args.data_dir),
        "--out-dir",
        str(taxonomy_dir),
    ]
    if args.quick:
        taxonomy_args.append("--quick")
    run_taxonomy(taxonomy_args)

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "quick": args.quick,
        "git_commit": git_commit,
        "commands": {
            "no_filter": ["uv", "run", "python", "scripts/no_filter_analysis.py", *no_filter_args],
            "synchronous_ablation": [
                "uv",
                "run",
                "python",
                "scripts/synchronous_ablation.py",
                *sync_args,
            ],
            "ranking_stability": [
                "uv",
                "run",
                "python",
                "scripts/ranking_stability.py",
                *ranking_args,
            ],
            "te_null": ["uv", "run", "python", "scripts/te_null_analysis.py", *te_args],
            "phenotype_taxonomy": [
                "uv",
                "run",
                "python",
                "scripts/phenotype_taxonomy.py",
                *taxonomy_args,
            ],
        },
        "outputs": {
            "no_filter": {
                "json": str(no_filter_dir / "summary.json"),
                "csv": str(no_filter_dir / "summary.csv"),
            },
            "synchronous_ablation": {
                "json": str(sync_dir / "summary.json"),
                "csv": str(sync_dir / "summary.csv"),
            },
            "ranking_stability": {
                "json": str(ranking_dir / "summary.json"),
                "csv": str(ranking_dir / "summary.csv"),
            },
            "te_null": {
                "json": str(te_dir / "summary.json"),
                "csv": str(te_dir / "summary.csv"),
            },
            "phenotype_taxonomy": {
                "json": str(taxonomy_dir / "taxonomy.json"),
                "csv": str(taxonomy_dir / "taxonomy.csv"),
            },
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
