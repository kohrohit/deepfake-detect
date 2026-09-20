"""Exception hierarchy. One catchable root for everything this package raises."""
from __future__ import annotations


class DfdError(Exception):
    """Base for every error raised by the dfd package."""


class InvalidInput(DfdError):
    """A caller supplied an argument that cannot be processed."""


class ResourceLimitExceeded(DfdError):
    """Input exceeded a configured decode or size limit."""
