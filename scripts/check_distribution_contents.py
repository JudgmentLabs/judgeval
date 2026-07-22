#!/usr/bin/env python3
"""Validate that built distributions contain only public package files."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


GENERATED_MODULES = {
    "judgeval/jql/_generated_contract.py",
    "judgeval/jql/_generated_transport.py",
}


def archive_names(path: Path) -> set[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
        roots = {name.split("/", 1)[0] for name in names}
        if len(roots) != 1:
            raise RuntimeError(f"sdist must have one archive root, found {roots}")
        root = roots.pop()
        return {name.removeprefix(f"{root}/") for name in names if name != root}
    raise ValueError(f"Unsupported distribution archive: {path}")


def unexpected_names(path: Path, names: set[str]) -> set[str]:
    if path.suffix == ".whl":
        return {
            name
            for name in names
            if not name.startswith("judgeval/") and ".dist-info/" not in name
        }
    allowed_files = {
        ".gitignore",
        "LICENSE.md",
        "README.md",
        "hatch_build.py",
        "pyproject.toml",
        "PKG-INFO",
    }
    return {
        name
        for name in names
        if name not in allowed_files
        and name != "src"
        and not name.startswith("src/judgeval")
    }


def main() -> None:
    distributions = sorted(
        [*Path("dist").glob("*.whl"), *Path("dist").glob("*.tar.gz")]
    )
    if not distributions:
        raise RuntimeError("No distributions found in dist/")

    errors = []
    for path in distributions:
        names = archive_names(path)
        unexpected = sorted(unexpected_names(path, names))
        if unexpected:
            errors.append(f"{path}: unexpected files: {', '.join(unexpected)}")

        generated = (
            GENERATED_MODULES
            if path.suffix == ".whl"
            else {f"src/{name}" for name in GENERATED_MODULES}
        )
        missing = sorted(generated - names)
        if missing:
            errors.append(f"{path}: missing generated files: {', '.join(missing)}")

    if errors:
        raise RuntimeError("Invalid distribution contents:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
