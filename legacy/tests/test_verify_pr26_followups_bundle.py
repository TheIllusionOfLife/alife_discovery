import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_pr26_followups_bundle import main as verify_main
from scripts.verify_pr26_followups_bundle import verify_bundle


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_bundle(tmp_path: Path) -> Path:
    base = tmp_path / "bundle"
    manifest = {
        "outputs": {
            "a": {
                "json": "no_filter/summary.json",
                "csv": "no_filter/summary.csv",
            }
        },
        "zenodo": {
            "files": [
                {
                    "name": "summary.json",
                    "relative_path": "no_filter/summary.json",
                    "sha256": "placeholder",
                },
                {
                    "name": "summary.csv",
                    "relative_path": "no_filter/summary.csv",
                    "sha256": "placeholder",
                },
            ]
        },
    }
    _write(base / "no_filter" / "summary.json", '{"ok": true}\n')
    _write(base / "no_filter" / "summary.csv", "k,v\nok,1\n")
    _write(base / "manifest.json", json.dumps(manifest, indent=2))

    payload = json.loads((base / "manifest.json").read_text())
    payload["zenodo"]["files"][0]["sha256"] = _sha(base / "no_filter" / "summary.json")
    payload["zenodo"]["files"][1]["sha256"] = _sha(base / "no_filter" / "summary.csv")
    (base / "manifest.json").write_text(json.dumps(payload, indent=2))

    checksums = "\n".join(
        [
            f"{_sha(base / 'manifest.json')}  manifest.json",
            f"{_sha(base / 'no_filter' / 'summary.json')}  no_filter/summary.json",
            f"{_sha(base / 'no_filter' / 'summary.csv')}  no_filter/summary.csv",
        ]
    )
    _write(base / "checksums.sha256", checksums + "\n")
    return base


def test_verify_bundle_ok(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    ok, errors = verify_bundle(bundle)
    assert ok
    assert errors == []


def test_verify_bundle_fails_on_checksum_mismatch(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    (bundle / "no_filter" / "summary.csv").write_text("k,v\nok,2\n")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("Checksum mismatch" in err for err in errors)


def test_verify_bundle_fails_on_unexpected_checksum_entry(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    with (bundle / "checksums.sha256").open("a") as handle:
        handle.write("abc  extra.json\n")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("unexpected entries" in err for err in errors)


def test_verify_bundle_fails_on_missing_manifest_output(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    (bundle / "no_filter" / "summary.json").unlink()
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("skipped entries" in err for err in errors)


def test_verify_bundle_fails_on_zenodo_hash_mismatch(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    payload = json.loads((bundle / "manifest.json").read_text())
    payload["zenodo"]["files"][0]["sha256"] = "0" * 64
    (bundle / "manifest.json").write_text(json.dumps(payload, indent=2))
    checksum_lines = (bundle / "checksums.sha256").read_text().splitlines()
    updated_lines: list[str] = []
    for line in checksum_lines:
        if line.endswith("manifest.json"):
            updated_lines.append(f"{_sha(bundle / 'manifest.json')}  manifest.json")
        else:
            updated_lines.append(line)
    (bundle / "checksums.sha256").write_text("\n".join(updated_lines) + "\n")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("Zenodo hash mismatch" in err for err in errors)
    assert not any("Checksum mismatch for manifest.json" in err for err in errors)


def test_verify_bundle_fails_on_checksum_path_traversal(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    with (bundle / "checksums.sha256").open("a") as handle:
        handle.write(f"{'0' * 64}  ../../etc/passwd\n")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("escapes bundle directory" in err for err in errors)


def test_verify_bundle_fails_on_zenodo_path_traversal(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    payload = json.loads((bundle / "manifest.json").read_text())
    payload["zenodo"]["files"].append(
        {
            "name": "escape",
            "relative_path": "../../etc/passwd",
            "sha256": "0" * 64,
        }
    )
    (bundle / "manifest.json").write_text(json.dumps(payload, indent=2))
    checksum_lines = (bundle / "checksums.sha256").read_text().splitlines()
    updated_lines: list[str] = []
    for line in checksum_lines:
        if line.endswith("manifest.json"):
            updated_lines.append(f"{_sha(bundle / 'manifest.json')}  manifest.json")
        else:
            updated_lines.append(line)
    (bundle / "checksums.sha256").write_text("\n".join(updated_lines) + "\n")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("zenodo relative_path escapes bundle directory" in err for err in errors)
    assert not any("Checksum mismatch for manifest.json" in err for err in errors)


def test_verify_bundle_missing_files(tmp_path: Path) -> None:
    ok, errors = verify_bundle(tmp_path / "not_there")
    assert not ok
    assert any("Missing manifest" in err for err in errors)
    assert all(isinstance(err, str) for err in errors)


def test_verify_bundle_rejects_manifest_directory(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle_dir_manifest"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").mkdir()
    (bundle / "checksums.sha256").write_text("")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("Missing manifest" in err for err in errors)


def test_verify_bundle_rejects_checksums_directory(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle_dir_checksums"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}")
    (bundle / "checksums.sha256").mkdir()
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("Missing checksums" in err for err in errors)


def test_verify_bundle_fails_on_malformed_manifest(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    (bundle / "manifest.json").write_text("{bad json")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("Invalid JSON in manifest" in err for err in errors)


def test_verify_bundle_fails_on_malformed_checksum_line(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    (bundle / "checksums.sha256").write_text("not-a-valid-checksum-line\n")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("Invalid checksums file" in err for err in errors)


def test_main_exits_on_failure(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(SystemExit) as exc_info:
        verify_main(["--followup-dir", str(missing)])
    assert exc_info.value.code == 1


def test_main_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = _build_bundle(tmp_path)
    verify_main(["--followup-dir", str(bundle)])
    captured = capsys.readouterr()
    assert "Bundle verification: OK" in captured.out


def test_verify_bundle_reports_missing_zenodo_checksums_entry_path(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    payload = json.loads((bundle / "manifest.json").read_text())
    payload["zenodo"]["files"].append(
        {
            "name": "checksums.sha256",
            "relative_path": "missing/checksums.sha256",
            "sha256": "0" * 64,
        }
    )
    (bundle / "manifest.json").write_text(json.dumps(payload, indent=2))
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("zenodo.files entry points to missing file" in err for err in errors)
    assert any("missing/checksums.sha256" in err for err in errors)


def test_verify_bundle_zenodo_missing_sha256(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    payload = json.loads((bundle / "manifest.json").read_text())
    del payload["zenodo"]["files"][0]["sha256"]
    (bundle / "manifest.json").write_text(json.dumps(payload, indent=2))
    checksum_lines = (bundle / "checksums.sha256").read_text().splitlines()
    updated_lines: list[str] = []
    for line in checksum_lines:
        if line.endswith("manifest.json"):
            updated_lines.append(f"{_sha(bundle / 'manifest.json')}  manifest.json")
        else:
            updated_lines.append(line)
    (bundle / "checksums.sha256").write_text("\n".join(updated_lines) + "\n")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("zenodo.files missing sha256" in err for err in errors)


def test_verify_bundle_zenodo_nonstring_sha256(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    payload = json.loads((bundle / "manifest.json").read_text())
    payload["zenodo"]["files"][0]["sha256"] = 123
    (bundle / "manifest.json").write_text(json.dumps(payload, indent=2))
    checksum_lines = (bundle / "checksums.sha256").read_text().splitlines()
    updated_lines: list[str] = []
    for line in checksum_lines:
        if line.endswith("manifest.json"):
            updated_lines.append(f"{_sha(bundle / 'manifest.json')}  manifest.json")
        else:
            updated_lines.append(line)
    (bundle / "checksums.sha256").write_text("\n".join(updated_lines) + "\n")
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("zenodo.files sha256 is not string" in err for err in errors)
