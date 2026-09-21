from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..types import Context, Sample


class IngestAdapter(Protocol):
    def __call__(self, path: str | Path, context: Context) -> Sample: ...
