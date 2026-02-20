import subprocess

import pytest

from scripts.reproduce_pr26_followups import main as reproduce_main


def test_reproduce_quick_invokes_expected_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool, env=None) -> subprocess.CompletedProcess[str]:
        assert check is True
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("scripts.reproduce_pr26_followups.subprocess.run", _fake_run)
    reproduce_main(["--mode", "quick"])
    assert calls[0][:4] == ["uv", "run", "python", "scripts/run_pr26_followups.py"]
    assert "--quick" in calls[0]
    assert calls[1][:4] == ["uv", "run", "python", "scripts/verify_pr26_followups_bundle.py"]
    assert calls[2][:4] == ["uv", "run", "python", "scripts/render_pr26_followups_tex.py"]


def test_reproduce_publish_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ZENODO_TOKEN"):
        reproduce_main(["--publish"])


def test_reproduce_with_paper_runs_tectonic(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool, env=None) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("scripts.reproduce_pr26_followups.subprocess.run", _fake_run)
    reproduce_main(["--mode", "quick", "--with-paper"])
    assert ["tectonic", "paper/main.tex"] in calls
    assert ["tectonic", "paper/supplementary.tex"] in calls
