"""One-command reproducibility runner for PR26 follow-up artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce PR26 follow-up outputs")
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage_d"))
    parser.add_argument("--followup-dir", type=Path, default=Path("data/post_hoc/pr26_followups"))
    parser.add_argument(
        "--render-output", type=Path, default=Path("paper/generated/pr26_followups.tex")
    )
    parser.add_argument("--with-paper", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--zenodo-sandbox", action="store_true")
    args = parser.parse_args(argv)
    if args.publish and not os.environ.get("ZENODO_TOKEN"):
        raise RuntimeError("--publish requested but ZENODO_TOKEN is not set")

    run_args = [
        "uv",
        "run",
        "python",
        "scripts/run_pr26_followups.py",
        "--data-dir",
        str(args.data_dir),
        "--out-dir",
        str(args.followup_dir),
    ]
    if args.mode == "quick":
        run_args.append("--quick")
    _run(run_args)

    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/verify_pr26_followups_bundle.py",
            "--followup-dir",
            str(args.followup_dir),
        ]
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/render_pr26_followups_tex.py",
            "--followup-dir",
            str(args.followup_dir),
            "--output",
            str(args.render_output),
        ]
    )

    if args.with_paper:
        _run(["tectonic", "paper/main.tex"])
        _run(["tectonic", "paper/supplementary.tex"])

    if args.publish:
        env = dict(os.environ)
        publish_args = [
            "uv",
            "run",
            "python",
            "scripts/publish_pr26_followups_zenodo.py",
            "--followup-dir",
            str(args.followup_dir),
            "--manifest",
            str(args.followup_dir / "manifest.json"),
            "--publish",
        ]
        if args.zenodo_sandbox:
            publish_args.append("--sandbox")
        _run(publish_args, env=env)

    print("Reproduction pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        raise
