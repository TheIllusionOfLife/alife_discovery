import json
import os
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
    publish_called = False

    def _fake_request_json(
        method: str, url: str, token: str, payload: object | None = None
    ) -> dict:
        nonlocal publish_called
        calls.append((method, url))
        if method == "POST" and "actions/publish" in url:
            publish_called = True
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
                "metadata": {"prereserve_doi": {"doi": "10.5072/zenodo.123"}},
            }
        if method == "PUT" and "depositions/123" in url:
            return {"id": 123}
        raise AssertionError(f"Unexpected request: {method} {url}")

    upload_names: list[str] = []

    def _fake_upload_file(url: str, token: str, file_path: Path) -> dict:
        if publish_called and file_path.name == "manifest.json":
            raise AssertionError("manifest upload must happen before publish")
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
    checksums_entry = next(
        entry for entry in updated["zenodo"]["files"] if entry["name"] == "checksums.sha256"
    )
    assert "sha256" not in checksums_entry
    assert "manifest.json" in upload_names
    manifest_index = upload_names.index("manifest.json")
    assert manifest_index > 0
    assert all(name != "manifest.json" for name in upload_names[:manifest_index])
    assert all(name != "manifest.json" for name in upload_names[manifest_index + 1 :])
    assert upload_names[0] == "checksums.sha256"
    publish_idx = next(
        idx
        for idx, (method, url) in enumerate(calls)
        if method == "POST" and "actions/publish" in url
    )
    metadata_idx = next(
        idx
        for idx, (method, url) in enumerate(calls)
        if method == "PUT" and "depositions/123" in url
    )
    create_idx = next(
        idx for idx, (method, url) in enumerate(calls) if method == "POST" and "depositions" in url
    )
    assert create_idx < publish_idx
    assert metadata_idx < publish_idx
    assert any(method == "POST" and "depositions" in url for method, url in calls)


def test_publish_zenodo_uploads_only_manifest_declared_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    followup_dir = tmp_path / "followups"
    manifest_path = followup_dir / "manifest.json"
    summary_json = followup_dir / "no_filter" / "summary.json"
    summary_csv = followup_dir / "no_filter" / "summary.csv"
    stray_json = followup_dir / "no_filter" / "filtered" / "phase_1" / "rules" / "rule.json"
    _write_file(summary_json, '{"ok": true}\n')
    _write_file(summary_csv, "k,v\nok,1\n")
    _write_file(stray_json, '{"heavy": true}\n')
    _write_file(followup_dir / "checksums.sha256", "abc  no_filter/summary.json\n")
    _write_file(
        manifest_path,
        json.dumps(
            {
                "schema_version": "1.0",
                "outputs": {
                    "no_filter": {
                        "json": str(summary_json),
                        "csv": str(summary_csv),
                    }
                },
            }
        ),
    )
    monkeypatch.setenv("ZENODO_TOKEN", "token")

    def _fake_request_json(
        method: str, url: str, token: str, payload: object | None = None
    ) -> dict:
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
                "metadata": {"prereserve_doi": {"doi": "10.5072/zenodo.123"}},
            }
        if method == "PUT" and "depositions/123" in url:
            return {"id": 123}
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

    assert "checksums.sha256" in upload_names
    assert "summary.json" in upload_names
    assert "summary.csv" in upload_names
    assert "rule.json" not in upload_names


def test_publish_zenodo_resolves_relative_paths_from_manifest_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    followup_dir = tmp_path / "followups"
    manifest_path = followup_dir / "manifest.json"
    summary_json = followup_dir / "no_filter" / "summary.json"
    summary_csv = followup_dir / "no_filter" / "summary.csv"
    _write_file(summary_json, '{"ok": true}\n')
    _write_file(summary_csv, "k,v\nok,1\n")
    _write_file(followup_dir / "checksums.sha256", "abc  no_filter/summary.json\n")
    _write_file(
        manifest_path,
        json.dumps(
            {
                "schema_version": "1.0",
                "outputs": {
                    "no_filter": {
                        "json": "no_filter/summary.json",
                        "csv": "no_filter/summary.csv",
                    }
                },
            }
        ),
    )
    monkeypatch.setenv("ZENODO_TOKEN", "token")

    def _fake_request_json(
        method: str, url: str, token: str, payload: object | None = None
    ) -> dict:
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
                "metadata": {"prereserve_doi": {"doi": "10.5072/zenodo.123"}},
            }
        if method == "PUT" and "depositions/123" in url:
            return {"id": 123}
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

    original_cwd = Path.cwd()
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(other_cwd)
        publish_main(
            [
                "--followup-dir",
                str(followup_dir),
                "--manifest",
                str(manifest_path),
                "--publish",
            ]
        )
    finally:
        os.chdir(original_cwd)

    assert "checksums.sha256" in upload_names
    assert "summary.json" in upload_names
    assert "summary.csv" in upload_names
