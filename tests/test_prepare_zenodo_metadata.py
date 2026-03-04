from __future__ import annotations

from scripts.prepare_zenodo_metadata import _redact_argv


def test_redact_argv_replaces_sensitive_flag_values() -> None:
    argv = [
        "--experiment-name",
        "x",
        "--token",
        "abcd",
        "--api-key=mykey",
        "--output",
        "out.json",
    ]
    redacted = _redact_argv(argv)
    assert redacted[3] == "<redacted>"
    assert "<redacted>" in redacted[4]


def test_redact_argv_keeps_non_sensitive_values() -> None:
    argv = ["--steps", "500", "--seed-start", "0"]
    assert _redact_argv(argv) == argv
