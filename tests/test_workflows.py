"""Tests for release workflow wiring."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def test_sync_publish_checks_out_the_synchronized_commit() -> None:
    publish = (_ROOT / ".github/workflows/publish.yml").read_text()
    sync = (_ROOT / ".github/workflows/sync-library.yml").read_text()

    assert "checkout_ref:" in publish
    assert "ref: ${{ inputs.checkout_ref || github.sha }}" in publish
    assert "checkout_ref: main" in sync
