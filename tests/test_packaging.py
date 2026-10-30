"""The distributable wheel includes the complete application and dashboard."""

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def test_wheel_contains_subpackages_and_static_assets(tmp_path):
    if any(importlib.util.find_spec(name) is None for name in ("setuptools", "wheel")):
        pytest.skip("Offline wheel build backend unavailable: setuptools and wheel are required")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(root / "pyproject.toml", source)
    shutil.copytree(root / "orderflow_system", source / "orderflow_system",
                    ignore=shutil.ignore_patterns("__pycache__", "graphify-out", ".graphify"))
    wheels = tmp_path / "wheels"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(wheels)],
        cwd=source, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = wheels.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    expected = {
        "orderflow_system/data/infobar_feed.py", "orderflow_system/pipeline.py",
        "orderflow_system/dashboard/static/index.html", "orderflow_system/config/settings.py",
    }
    expected.update(str(path.relative_to(root)) for path in (root / "orderflow_system").rglob("__init__.py"))
    assert expected <= names
