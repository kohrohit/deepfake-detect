"""Exception hierarchy.

`DfdError` is meant to become one catchable root for everything this package
raises, but that is not true yet: as of Task 20, `fusion.py`, `calibration.py`,
`detectors/`, and `ingest/` still have 19 call sites that raise bare
`ValueError`/`RuntimeError` directly rather than a `DfdError` subclass.
Migrating those onto this hierarchy is an outstanding gap (it touches eight
already-completed tasks), not something this module can claim to have done by
existing. New code, and any exception this module or its siblings define, must
subclass `DfdError`.
"""
from __future__ import annotations


class DfdError(Exception):
    """Base for every error raised by the dfd package."""


class InvalidInput(DfdError):
    """A caller supplied an argument that cannot be processed."""


class ResourceLimitExceeded(DfdError):
    """Input exceeded a configured decode or size limit."""
