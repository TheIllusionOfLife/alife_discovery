import json
from pathlib import Path

from scripts.verify_pr26_followups_bundle import verify_bundle


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


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

    import hashlib

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

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
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("Zenodo hash mismatch" in err for err in errors)


def test_verify_bundle_missing_files(tmp_path: Path) -> None:
    ok, errors = verify_bundle(tmp_path / "not_there")
    assert not ok
    assert any("Missing manifest" in err for err in errors)
    assert all(isinstance(err, str) for err in errors)
