from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.upload_zenodo import _load_and_verify


def test_load_and_verify_rejects_path_escape(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("x")

    args = SimpleNamespace(no_verify_checksums=True)
    meta = {"artifacts": [{"path": str(outside)}]}

    with pytest.raises(SystemExit):
        _load_and_verify(args, meta, base)


def test_load_and_verify_accepts_in_base_file(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    f = base / "artifact.txt"
    f.write_text("hello")

    args = SimpleNamespace(no_verify_checksums=True)
    meta = {"artifacts": [{"path": "artifact.txt"}]}

    paths = _load_and_verify(args, meta, base)
    assert paths == [f.resolve()]
