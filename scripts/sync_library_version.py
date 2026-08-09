"""Sync the library dependency floor and bump the ComfyUI node patch version."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_NODE_VERSION = re.compile(r'^version = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)
_LIBRARY_DEPENDENCY = re.compile(
    r"(remove-ai-watermarks(?:\[[a-z0-9,-]+\])?>=)([0-9]+\.[0-9]+\.[0-9]+)"
)


def sync_text(
    pyproject: str, requirements: str, *, library_version: str
) -> tuple[str, str, bool]:
    """Return synchronized files and whether a node-version bump was needed."""
    if re.fullmatch(r"\d+\.\d+\.\d+", library_version) is None:
        raise ValueError("library_version must use X.Y.Z format")

    dependency_matches = _LIBRARY_DEPENDENCY.findall(pyproject)
    requirement_matches = _LIBRARY_DEPENDENCY.findall(requirements)
    if len(dependency_matches) != 1 or len(requirement_matches) != 1:
        raise ValueError("expected one library dependency in each package file")
    if (
        dependency_matches[0][1] == library_version
        and requirement_matches[0][1] == library_version
    ):
        return pyproject, requirements, False

    version_match = _NODE_VERSION.search(pyproject)
    if version_match is None:
        raise ValueError("expected one semantic node version in pyproject.toml")
    major, minor, patch = (int(value) for value in version_match.groups())
    node_version = f"{major}.{minor}.{patch + 1}"

    updated_project, version_count = _NODE_VERSION.subn(
        f'version = "{node_version}"',
        pyproject,
    )
    updated_project, project_dep_count = _LIBRARY_DEPENDENCY.subn(
        rf"\g<1>{library_version}",
        updated_project,
    )
    updated_requirements, requirements_count = _LIBRARY_DEPENDENCY.subn(
        rf"\g<1>{library_version}",
        requirements,
    )
    if (version_count, project_dep_count, requirements_count) != (1, 1, 1):
        raise ValueError("package metadata changed while synchronizing")
    return updated_project, updated_requirements, True


def main() -> None:
    """Synchronize the package files in place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-version", required=True)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    args = parser.parse_args()

    project_text = args.pyproject.read_text()
    requirements_text = args.requirements.read_text()
    updated_project, updated_requirements, changed = sync_text(
        project_text,
        requirements_text,
        library_version=args.library_version,
    )
    if changed:
        args.pyproject.write_text(updated_project)
        args.requirements.write_text(updated_requirements)
        log.info(
            "Updated the library floor to %s and bumped the node version",
            args.library_version,
        )
    else:
        log.info("Library floor is already %s", args.library_version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
