"""The CI config is itself tested: a gate nobody verifies is a gate that rots."""
import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_exists():
    assert (ROOT / ".github/workflows/ci.yml").is_file()


def test_ci_runs_every_gate():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    for gate in ("ruff", "mypy", "pytest"):
        assert gate in ci, f"CI does not run {gate}"


def test_ci_enforces_the_asset_registration_gate():
    """Confirms the CI step exists and calls the right function — nothing more.

    This gate cannot fail in CI today: `.github/workflows/ci.yml` calls
    `assert_all_assets_registered(..., allow_empty=True)`, and CI's weight
    files are gitignored, so the scan finds nothing to register and
    `allow_empty=True` lets that pass deliberately (see `asset_scan.py`).
    That design is correct — CI has no weights to check — but it means
    spec criterion 6 is enforced only where this function runs against a
    tree that actually has assets (e.g. locally, or in an environment with
    weights present), never by this CI job. Do not read this test, or a
    green CI run, as evidence that an unregistered weight file would be
    caught in CI — it would not.
    """
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
        [sys.executable, "-m", "ruff", "check", "src", "bench", "corpora", "training"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ci_lints_training_too():
    """`training/` carries the fitter (training/fit_blend.py) and is not
    itself installed or mypy-checked (mypy.ini scopes to src/dfd only,
    deliberately — see the plan's Global Constraints), but nothing exempts
    it from ruff. ci.yml's lint step must actually run over it, not just
    this test module."""
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    match = re.search(r"run:\s*ruff check ([^\n]+)", ci)
    assert match is not None, "could not find the ruff check invocation in ci.yml"
    assert match.group(1).split() == ["src", "bench", "corpora", "training"]


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
                    "pillow", "torch", "scikit-learn", "pyyaml", "packaging"):
        assert package in dev, f"{package} missing from requirements-dev.txt"


# --- Dependency drift ------------------------------------------------------
# This branch shipped "mypy --strict clean" and went red on its first CI run,
# because every dependency was a floor with no ceiling and CI resolved numpy
# 2.2.6 / opencv 5.0.0 / torch 2.14.0 against a local numpy 1.26.4 / opencv
# 4.10 / torch 2.4.1. Two real defects followed. These three tests keep the
# pins honest. Each asserts its parse found something first: a loop over an
# empty list passes against any file at all, which is the exact vacuous-test
# failure this plan hit roughly thirty times.

EXPECTED_DEV_PINS = 11
EXPECTED_FLOOR_PINS = 11
EXPECTED_PYPROJECT_DEPS = 6


def _dev_requirements() -> list[Requirement]:
    lines = (ROOT / "requirements-dev.txt").read_text().splitlines()
    return [Requirement(ln.strip()) for ln in lines
            if ln.strip() and not ln.lstrip().startswith("#")]


def _pyproject_requirements() -> list[Requirement]:
    """Parses the dependencies array without tomllib, which is 3.11+ while CI
    pins python 3.10."""
    text = (ROOT / "pyproject.toml").read_text()
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]",
                      text, re.DOTALL | re.MULTILINE)
    assert block is not None, "could not locate the dependencies array"
    return [Requirement(m) for m in re.findall(r'"([^"]+)"', block.group(1))]


def test_every_dev_requirement_is_exactly_pinned():
    """A floor in this file is how CI and local silently diverge. Exact pins are
    the only thing that made CI reproducible — ceilings would not have caught
    the torch 2.6 break, which landed in a minor release."""
    reqs = _dev_requirements()
    assert len(reqs) == EXPECTED_DEV_PINS, (
        f"parsed {len(reqs)} requirements, expected {EXPECTED_DEV_PINS}; "
        "update EXPECTED_DEV_PINS deliberately when adding a dependency")
    unpinned = [str(r) for r in reqs
                if {s.operator for s in r.specifier} != {"=="}]
    assert unpinned == [], f"requirements-dev.txt entries are not ==-pinned: {unpinned}"


def test_every_runtime_dependency_has_an_upper_bound():
    """An unbounded runtime dependency hands the next major release of numpy,
    opencv or torch a free pass into anyone who installs this package."""
    reqs = _pyproject_requirements()
    assert len(reqs) == EXPECTED_PYPROJECT_DEPS, (
        f"parsed {len(reqs)} dependencies, expected {EXPECTED_PYPROJECT_DEPS}")
    unbounded = [str(r) for r in reqs
                 if not any(s.operator in ("<", "<=") for s in r.specifier)]
    assert unbounded == [], f"pyproject dependencies lack an upper bound: {unbounded}"


def test_dev_pins_satisfy_the_pyproject_ranges():
    """CI runs `pip install -r requirements-dev.txt` and THEN `pip install -e .`.
    If a pin fell outside its pyproject range, that second command would quietly
    re-resolve it and the exact pin above would buy nothing — CI would once again
    be running versions nobody verified."""
    dev = {canonicalize_name(r.name): r for r in _dev_requirements()}
    project = _pyproject_requirements()
    assert len(project) == EXPECTED_PYPROJECT_DEPS

    checked = []
    for req in project:
        pin = dev.get(canonicalize_name(req.name))
        assert pin is not None, (
            f"{req.name} is a runtime dependency but is not pinned in "
            "requirements-dev.txt, so CI never fixes its version")
        version = str(next(iter(pin.specifier)).version)
        assert req.specifier.contains(version), (
            f"requirements-dev.txt pins {req.name}=={version}, which is outside "
            f"the pyproject range '{req.specifier}'. `pip install -e .` would "
            "re-resolve it and undo the pin.")
        checked.append(req.name)
    assert len(checked) == EXPECTED_PYPROJECT_DEPS


def _floor_requirements() -> list[Requirement]:
    lines = (ROOT / "requirements-floor.txt").read_text().splitlines()
    return [Requirement(ln.strip()) for ln in lines
            if ln.strip() and not ln.lstrip().startswith("#")]


def test_every_floor_requirement_is_exactly_pinned():
    """Same reasoning as the dev pins: a floor with no ceiling in this file
    would let the floor leg resolve upward and silently become a second copy
    of the pinned leg, testing nothing."""
    reqs = _floor_requirements()
    assert len(reqs) == EXPECTED_FLOOR_PINS, (
        f"parsed {len(reqs)} requirements, expected {EXPECTED_FLOOR_PINS}; "
        "update EXPECTED_FLOOR_PINS deliberately when adding a dependency")
    unpinned = [str(r) for r in reqs
                if {s.operator for s in r.specifier} != {"=="}]
    assert unpinned == [], f"requirements-floor.txt entries are not ==-pinned: {unpinned}"


def test_floor_pins_are_exactly_the_declared_pyproject_floors():
    """The point of the floor file is that `numpy>=1.26.4` in pyproject.toml is
    a claim someone can check. If the pin here merely *satisfies* the range
    instead of *being* its lower bound, the declared floor goes back to being
    an untested assertion while CI reports green against some higher version."""
    floor = {canonicalize_name(r.name): r for r in _floor_requirements()}
    project = _pyproject_requirements()
    assert len(project) == EXPECTED_PYPROJECT_DEPS

    checked = []
    for req in project:
        lower = [s for s in req.specifier if s.operator == ">="]
        assert len(lower) == 1, (
            f"{req.name} does not declare exactly one '>=' floor: '{req.specifier}'")
        pin = floor.get(canonicalize_name(req.name))
        assert pin is not None, (
            f"{req.name} is a runtime dependency but is not pinned in "
            "requirements-floor.txt, so its declared floor is never executed")
        pinned_version = Version(str(next(iter(pin.specifier)).version))
        declared_floor = Version(lower[0].version)
        assert pinned_version == declared_floor, (
            f"requirements-floor.txt pins {req.name}=={pinned_version}, but "
            f"pyproject.toml declares the floor as {declared_floor}. The floor "
            "leg would then exercise a version nobody declared, and the "
            "declared floor would stay untested.")
        checked.append(req.name)
    assert len(checked) == EXPECTED_PYPROJECT_DEPS


def test_the_two_requirement_files_differ_only_in_runtime_dependencies():
    """Two things, both load-bearing. Tooling identical: if ruff or mypy also
    moved between the legs, a red floor leg would not say whether the runtime
    floor or the tool broke it. And at least one runtime version must actually
    differ, or the floor leg is a duplicate of the pinned leg — green, costing
    CI minutes, and proving nothing."""
    dev = {canonicalize_name(r.name): str(next(iter(r.specifier)).version)
           for r in _dev_requirements()}
    floor = {canonicalize_name(r.name): str(next(iter(r.specifier)).version)
             for r in _floor_requirements()}
    assert set(dev) == set(floor), (
        "the two requirement files name different packages: "
        f"dev-only={sorted(set(dev) - set(floor))}, "
        f"floor-only={sorted(set(floor) - set(dev))}")

    runtime = {canonicalize_name(r.name) for r in _pyproject_requirements()}
    tools_that_moved = {name: (dev[name], floor[name]) for name in dev
                        if name not in runtime and dev[name] != floor[name]}
    assert tools_that_moved == {}, (
        "the dev tooling is not held constant across the two legs, so a red "
        f"floor leg would be ambiguous: {tools_that_moved}")

    moved = {name for name in runtime if dev[name] != floor[name]}
    assert moved, (
        "every runtime dependency is pinned identically in both files, so the "
        "floor leg re-runs the pinned leg and exercises no floor at all")


def test_ci_runs_the_gates_at_both_ends_of_every_range():
    """Before this, both ends were tested only because two machines happened to
    sit at opposite ends, and CI ran the upper end alone — which is how a break
    that only appears under one resolution stays invisible until someone's
    laptop is replaced."""
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "matrix:" in ci, "CI defines no matrix, so it runs one resolution only"
    for name in ("requirements-dev.txt", "requirements-floor.txt"):
        stem = name.removeprefix("requirements-").removesuffix(".txt")
        assert stem in ci, f"CI never installs {name}, so that end is unexercised"
    assert "fail-fast: false" in ci, (
        "without fail-fast: false a red pinned leg cancels the floor leg, and "
        "the floor result is lost exactly when it is most interesting")

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
