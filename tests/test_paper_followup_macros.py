from pathlib import Path


def test_supplementary_loads_pr26_followup_macros() -> None:
    supplementary = Path("paper/supplementary.tex").read_text()
    assert "\\IfFileExists{generated/pr26_followups.tex}" in supplementary
    assert "\\input{generated/pr26_followups.tex}" in supplementary
    assert "\\input{generated/pr26_followups.defaults.tex}" in supplementary
