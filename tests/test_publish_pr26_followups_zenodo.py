import json
from pathlib import Path

import pytest

from scripts.publish_pr26_followups_zenodo import main as publish_main


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_publish_zenodo_requires_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    followup_dir = tmp_path / "followups"
    _write_file(followup_dir / "manifest.json", json.dumps({"schema_version": "1.0"}))
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ZENODO_TOKEN"):
        publish_main(
            [
                "--followup-dir",
                str(followup_dir),
                "--manifest",
                str(followup_dir / "manifest.json"),
                "--publish",
            ]
        )


def test_publish_zenodo_updates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    followup_dir = tmp_path / "followups"
    manifest_path = followup_dir / "manifest.json"
    payload_path = followup_dir / "no_filter" / "summary.json"
    _write_file(payload_path, '{"ok": true}\n')
    _write_file(manifest_path, json.dumps({"schema_version": "1.0", "outputs": {}}))
    _write_file(followup_dir / "checksums.sha256", "")
    monkeypatch.setenv("ZENODO_TOKEN", "token")

    calls: list[tuple[str, str]] = []

    def _fake_request_json(
        method: str, url: str, token: str, payload: object | None = None
    ) -> dict:
        calls.append((method, url))
        if method == "POST" and "actions/publish" in url:
            return {
                "doi": "10.5072/zenodo.123",
                "links": {"html": "https://zenodo.example/records/123"},
            }
        if method == "POST" and "depositions" in url:
            return {
                "id": 123,
                "links": {
                    "bucket": "https://zenodo.example/api/files/bucket-123",
                    "html": "https://zenodo.example/records/123",
                    "publish": "https://zenodo.example/api/deposit/depositions/123/actions/publish",
                },
            }
        raise AssertionError(f"Unexpected request: {method} {url}")

    upload_names: list[str] = []

    def _fake_upload_file(url: str, token: str, file_path: Path) -> dict:
        upload_names.append(file_path.name)
        return {"download": f"https://zenodo.example/files/{file_path.name}"}

    monkeypatch.setattr(
        "scripts.publish_pr26_followups_zenodo._request_json",
        _fake_request_json,
    )
    monkeypatch.setattr(
        "scripts.publish_pr26_followups_zenodo._upload_file",
        _fake_upload_file,
    )

    publish_main(
        [
            "--followup-dir",
            str(followup_dir),
            "--manifest",
            str(manifest_path),
            "--publish",
        ]
    )

    updated = json.loads(manifest_path.read_text())
    assert updated["zenodo"]["doi"] == "10.5072/zenodo.123"
    assert updated["zenodo"]["record_url"] == "https://zenodo.example/records/123"
    assert updated["zenodo"]["deposit_id"] == 123
    names = [entry["name"] for entry in updated["zenodo"]["files"]]
    assert "checksums.sha256" in names
    assert "manifest.json" not in names
    assert "manifest.json" in upload_names
    assert any(method == "POST" and "depositions" in url for method, url in calls)
