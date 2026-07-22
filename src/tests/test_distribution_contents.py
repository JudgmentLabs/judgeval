from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType


def load_checker() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "check_distribution_contents.py"
    )
    spec = importlib.util.spec_from_file_location("check_distribution_contents", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_wheel(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def valid_wheel_files() -> dict[str, bytes]:
    dist_info = "judgeval-1.0.0.dist-info"
    return {
        "judgeval/__init__.py": b"",
        f"{dist_info}/METADATA": b"",
        f"{dist_info}/WHEEL": b"",
        f"{dist_info}/entry_points.txt": b"",
        f"{dist_info}/licenses/LICENSE.md": b"",
        f"{dist_info}/RECORD": b"",
    }


def test_exact_manifest_rejects_untracked_package_files(tmp_path: Path) -> None:
    checker = load_checker()
    wheel = tmp_path / "judgeval-1.0.0-py3-none-any.whl"
    files = valid_wheel_files()
    files["judgeval/private_source.py"] = b"secret"
    write_wheel(wheel, files)

    errors = checker.distribution_errors(wheel, {"src/judgeval/__init__.py"})

    assert errors == ["unexpected files: judgeval/private_source.py"]


def test_archive_content_scan_rejects_private_marker(tmp_path: Path) -> None:
    checker = load_checker()
    wheel = tmp_path / "judgeval-1.0.0-py3-none-any.whl"
    files = valid_wheel_files()
    files["judgeval/__init__.py"] = checker.PRIVATE_SOURCE_MARKER
    write_wheel(wheel, files)

    errors = checker.distribution_errors(wheel, {"src/judgeval/__init__.py"})

    assert errors == ["private-source marker found in: judgeval/__init__.py"]
