#!/usr/bin/env python3
"""Validate built distributions against the tracked public package manifest."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


PRIVATE_SOURCE_MARKER = b"PRIVATE MONOREPO CONTENT"
MAX_ARCHIVE_FILES = 1_000
MAX_ARCHIVE_BYTES = 5 * 1024 * 1024
MAX_UNPACKED_BYTES = 10 * 1024 * 1024
SDIST_METADATA = {
    ".gitignore",
    "LICENSE.md",
    "README.md",
    "hatch_build.py",
    "pyproject.toml",
    "PKG-INFO",
}
WHEEL_METADATA = {
    "METADATA",
    "WHEEL",
    "entry_points.txt",
    "licenses/LICENSE.md",
    "RECORD",
}


def tracked_package_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "src/judgeval"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {name for name in result.stdout.split("\0") if name}


def archive_entries(path: Path) -> list[tuple[str, bytes]]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return [
                (info.filename, archive.read(info))
                for info in archive.infolist()
                if not info.is_dir()
            ]
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            roots = {member.name.split("/", 1)[0] for member in members}
            if len(roots) != 1:
                raise RuntimeError(f"sdist must have one archive root, found {roots}")
            root = roots.pop()
            special = [
                member.name
                for member in members
                if not member.isfile() and not member.isdir()
            ]
            if special:
                raise RuntimeError(
                    f"sdist contains links or special entries: {', '.join(special)}"
                )
            return [
                (
                    member.name.removeprefix(f"{root}/"),
                    archive.extractfile(member).read(),
                )
                for member in members
                if member.isfile()
            ]
    raise ValueError(f"Unsupported distribution archive: {path}")


def archive_files(path: Path) -> dict[str, bytes]:
    entries = archive_entries(path)
    files = dict(entries)
    if len(files) != len(entries):
        raise RuntimeError("archive contains duplicate entries")
    return files


def expected_names(path: Path, names: set[str], tracked: set[str]) -> set[str]:
    if path.suffix == ".whl":
        roots = {name.split("/", 1)[0] for name in names}
        dist_info_roots = {root for root in roots if root.endswith(".dist-info")}
        if len(dist_info_roots) != 1:
            raise RuntimeError(
                f"wheel must have one dist-info root, found {dist_info_roots}"
            )
        dist_info = dist_info_roots.pop()
        package_files = {name.removeprefix("src/") for name in tracked}
        metadata = {f"{dist_info}/{name}" for name in WHEEL_METADATA}
        return package_files | metadata
    return tracked | SDIST_METADATA


def distribution_errors(path: Path, tracked: set[str]) -> list[str]:
    files = archive_files(path)
    names = set(files)
    expected = expected_names(path, names, tracked)
    errors = []

    unexpected = sorted(names - expected)
    if unexpected:
        errors.append(f"unexpected files: {', '.join(unexpected)}")
    missing = sorted(expected - names)
    if missing:
        errors.append(f"missing files: {', '.join(missing)}")
    if len(files) > MAX_ARCHIVE_FILES:
        errors.append(f"contains {len(files)} files; limit is {MAX_ARCHIVE_FILES}")
    archive_bytes = path.stat().st_size
    if archive_bytes > MAX_ARCHIVE_BYTES:
        errors.append(f"archive is {archive_bytes} bytes; limit is {MAX_ARCHIVE_BYTES}")
    unpacked_bytes = sum(len(content) for content in files.values())
    if unpacked_bytes > MAX_UNPACKED_BYTES:
        errors.append(
            f"unpacked size is {unpacked_bytes} bytes; limit is {MAX_UNPACKED_BYTES}"
        )

    marked = sorted(
        name for name, content in files.items() if PRIVATE_SOURCE_MARKER in content
    )
    if marked:
        errors.append(f"private-source marker found in: {', '.join(marked)}")
    return errors


def plant_marker() -> None:
    marker = Path(".jql-source/judgment-mono/PRIVATE_SOURCE_MARKER")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(PRIVATE_SOURCE_MARKER)


def main() -> None:
    distributions = sorted(
        [*Path("dist").glob("*.whl"), *Path("dist").glob("*.tar.gz")]
    )
    if not distributions:
        raise RuntimeError("No distributions found in dist/")

    tracked = tracked_package_files()
    errors = [
        f"{path}: {error}"
        for path in distributions
        for error in distribution_errors(path, tracked)
    ]
    if errors:
        raise RuntimeError("Invalid distribution contents:\n" + "\n".join(errors))


if __name__ == "__main__":
    if "--plant" in sys.argv:
        plant_marker()
    else:
        main()
