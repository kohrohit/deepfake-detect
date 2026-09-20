"""The CI config is itself tested: a gate nobody verifies is a gate that rots."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_exists():
    assert (ROOT / ".github/workflows/ci.yml").is_file()


def test_ci_runs_every_gate():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    for gate in ("ruff", "mypy", "pytest"):
        assert gate in ci, f"CI does not run {gate}"


def test_ci_enforces_the_asset_registration_gate():
    """Spec criterion 6 is only real if CI fails on an unregistered weight file."""
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "assert_all_assets_registered" in ci or "asset_scan" in ci


def test_mypy_is_configured_strict():
    cfg = (ROOT / "mypy.ini").read_text()
    assert "strict = True" in cfg or "strict=True" in cfg


def test_ruff_actually_passes():
    """Runs the gate. Asserting the config file contains "E722" proves a
    string is in a file, not that the tree is clean — the suite would go
    green here while CI went red on the first push."""
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "bench", "corpora"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_mypy_strict_actually_passes():
    """Runs the gate. ~1s warm, ~41s cold — worth it for a gate that
    otherwise rots silently."""
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", "mypy.ini"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ruff_bans_silent_exception_handling():
    """Global constraint: a swallowed error in a fraud detector is an approved fraud."""
    cfg = (ROOT / "ruff.toml").read_text()
    # E722 = bare except, BLE = blind except, S110 = try/except/pass
    assert "E722" in cfg
    assert "BLE" in cfg or "S110" in cfg


def test_the_declared_dependencies_cover_what_is_imported():
    """CI installs from these files; a missing entry is a red pipeline on a
    clean runner and nothing at all locally, where the package is present."""
    dev = (ROOT / "requirements-dev.txt").read_text().lower()
    for package in ("pytest", "ruff", "mypy", "numpy", "opencv-python-headless",
                    "pillow", "torch", "scikit-learn", "pyyaml"):
        assert package in dev, f"{package} missing from requirements-dev.txt"


def test_pyproject_declares_runtime_dependencies():
    """pyproject declared none at all, so `pip install .` produced a package
    that imports numpy, opencv, torch and Pillow and depends on none of them."""
    cfg = (ROOT / "pyproject.toml").read_text().lower()
    assert "dependencies" in cfg
    for package in ("numpy", "opencv", "pillow"):
        assert package in cfg, f"{package} not declared in pyproject.toml"


def test_no_bare_except_anywhere_in_src():
    """Enforced here too, so the rule holds even if ruff config drifts."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip() == "except:":
                offenders.append(f"{path}:{n}")
    assert offenders == [], f"bare except found: {offenders}"


def test_no_print_statements_in_src():
    """Global constraint: structured logging, never print."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip().startswith("print("):
                offenders.append(f"{path}:{n}")
    assert offenders == [], f"print() found in src: {offenders}"
