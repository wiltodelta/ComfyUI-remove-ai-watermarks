"""Tests for automatic library dependency synchronization."""

from __future__ import annotations

import pytest

from scripts.sync_library_version import sync_text

_PROJECT = """\
[project]
version = "0.1.2"
dependencies = ["remove-ai-watermarks[qwen-zimage]>=0.19.0"]
"""
_REQUIREMENTS = "remove-ai-watermarks[qwen-zimage]>=0.19.0\n"


def test_sync_text_updates_both_floors_and_bumps_patch() -> None:
    project, requirements, changed = sync_text(
        _PROJECT,
        _REQUIREMENTS,
        library_version="0.20.0",
    )

    assert changed
    assert 'version = "0.1.3"' in project
    assert "remove-ai-watermarks[qwen-zimage]>=0.20.0" in project
    assert requirements == "remove-ai-watermarks[qwen-zimage]>=0.20.0\n"


def test_sync_text_is_idempotent() -> None:
    project = _PROJECT.replace("0.19.0", "0.20.0")
    requirements = _REQUIREMENTS.replace("0.19.0", "0.20.0")

    new_project, new_requirements, changed = sync_text(
        project,
        requirements,
        library_version="0.20.0",
    )

    assert not changed
    assert new_project == project
    assert new_requirements == requirements


@pytest.mark.parametrize("version", ["", "0.20", "v0.20.0", "latest"])
def test_sync_text_rejects_non_release_versions(version: str) -> None:
    with pytest.raises(ValueError, match="X.Y.Z"):
        sync_text(_PROJECT, _REQUIREMENTS, library_version=version)
