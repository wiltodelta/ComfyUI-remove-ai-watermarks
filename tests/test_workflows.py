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


def test_publish_action_preserves_the_prepared_checkout() -> None:
    publish = (_ROOT / ".github/workflows/publish.yml").read_text()

    assert "skip_checkout: true" in publish


def test_publish_verifies_registry_state_after_action_failure() -> None:
    publish = (_ROOT / ".github/workflows/publish.yml").read_text()

    assert "id: publish" in publish
    assert "continue-on-error: true" in publish
    assert "steps.publish.outcome == 'failure'" in publish
    assert "Verify published node version" in publish
    assert 'row["status"] == "NodeVersionStatusActive"' in publish


def test_sync_retries_pypi_release_visibility() -> None:
    sync = (_ROOT / ".github/workflows/sync-library.yml").read_text()

    assert "for attempt in $(seq 1 30)" in sync
    assert "Published release is not visible on PyPI yet" in sync
    assert "did not become visible on PyPI" in sync
