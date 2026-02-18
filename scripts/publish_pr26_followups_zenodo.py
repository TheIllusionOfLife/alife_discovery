"""Publish PR #26 follow-up bundle files to Zenodo and update manifest metadata.

Usage:
    uv run python scripts/publish_pr26_followups_zenodo.py \
      --followup-dir data/post_hoc/pr26_followups \
      --manifest data/post_hoc/pr26_followups/manifest.json \
      --publish
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib import error, parse, request


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(
    method: str,
    url: str,
    token: str,
    payload: object | None = None,
) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, method=method, data=data)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:  # pragma: no cover - error body asserted via RuntimeError text
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Zenodo request failed ({exc.code}): {body}") from exc


def _upload_file(url: str, token: str, file_path: Path) -> dict:
    req = request.Request(
        url=f"{url}/{parse.quote(file_path.name)}", method="PUT", data=file_path.read_bytes()
    )
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/octet-stream")
    try:
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:  # pragma: no cover - error body asserted via RuntimeError text
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Zenodo upload failed ({exc.code}): {body}") from exc


def _collect_upload_targets(followup_dir: Path, manifest_path: Path) -> list[Path]:
    targets: list[Path] = []
    checksums = followup_dir / "checksums.sha256"
    if checksums.exists():
        targets.append(checksums)
    if manifest_path.exists():
        targets.append(manifest_path)
    for path in sorted(followup_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".csv"}:
            continue
        if path in targets:
            continue
        targets.append(path)
    return targets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Upload follow-up bundle files to Zenodo")
    parser.add_argument("--followup-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deposit-id", type=int, default=None)
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org API")
    parser.add_argument("--publish", action="store_true", help="Publish deposition after upload")
    args = parser.parse_args(argv)

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise RuntimeError("ZENODO_TOKEN environment variable is required")

    followup_dir = Path(args.followup_dir)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    api_base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"

    if args.deposit_id is None:
        deposition = _request_json("POST", f"{api_base}/deposit/depositions", token, payload={})
    else:
        deposition = _request_json(
            "GET",
            f"{api_base}/deposit/depositions/{args.deposit_id}",
            token,
            payload=None,
        )

    bucket_url = deposition["links"]["bucket"]
    record_url = deposition["links"].get("html", "")
    deposit_id = int(deposition["id"])
    uploaded_files: list[dict[str, object]] = []

    for file_path in _collect_upload_targets(followup_dir, manifest_path):
        upload_resp = _upload_file(bucket_url, token, file_path)
        uploaded_files.append(
            {
                "name": file_path.name,
                "relative_path": str(file_path.relative_to(followup_dir)),
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
                "download_url": upload_resp.get("download", ""),
            }
        )

    doi = ""
    if args.publish:
        publish_url = deposition["links"]["publish"]
        published = _request_json("POST", publish_url, token, payload={})
        doi = str(published.get("doi", ""))
        record_url = published.get("links", {}).get("html", record_url)
    else:
        metadata_doi = deposition.get("metadata", {}).get("prereserve_doi", {}).get("doi")
        if metadata_doi:
            doi = str(metadata_doi)

    manifest["zenodo"] = {
        "sandbox": args.sandbox,
        "deposit_id": deposit_id,
        "doi": doi or "pending",
        "record_url": record_url,
        "files": uploaded_files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest["zenodo"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
