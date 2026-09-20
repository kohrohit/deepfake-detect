"""Loader for cached Reality Defender responses (spec §1.2).

These are the free head-to-head data: 24 results with per-model breakdowns,
already paid for. No quota is consumed by reading them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RDResult:
    cache_key: str
    verdict: str
    score: float
    model_scores: dict[str, float] = field(default_factory=dict)
    model_verdicts: dict[str, str] = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return len(self.model_scores)

    @property
    def n_manipulated(self) -> int:
        return sum(1 for v in self.model_verdicts.values() if v == "MANIPULATED")

    @property
    def max_model_score(self) -> float:
        return max(self.model_scores.values()) if self.model_scores else float("nan")


def load_rd_cache(root: str | Path) -> list[RDResult]:
    out: list[RDResult] = []
    for path in sorted(Path(root).glob("*/result.json")):
        try:
            d = json.loads(path.read_text())
            if not isinstance(d, dict):
                raise TypeError(
                    f"expected a JSON object, got {type(d).__name__}")
            models = d.get("models") or []
            if not isinstance(models, list):
                raise TypeError(
                    f"expected 'models' to be a list, got {type(models).__name__}")
            model_scores: dict[str, float] = {}
            model_verdicts: dict[str, str] = {}
            for m in models:
                if not isinstance(m, dict):
                    raise TypeError("model entry is not a JSON object")
                model_scores[m["name"]] = float(m["score"])
                model_verdicts[m["name"]] = m.get("verdict", "")
            result = RDResult(
                cache_key=path.parent.name,
                verdict=d.get("verdict", ""),
                score=float(d.get("score", float("nan"))),
                model_scores=model_scores,
                model_verdicts=model_verdicts,
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError,
                KeyError):
            # Skip a result that parses but is the wrong shape (a list, a
            # string, a non-numeric score, a model entry missing a field)
            # exactly as a decode/IO failure is skipped — one bad cache entry
            # must not take down the other 23.
            continue
        out.append(result)
    return out


def aggregate_is_max_like(results: list[RDResult], tolerance: float = 0.2) -> bool:
    """True if the vendor's aggregate tracks the maximum member score.

    Matters because a near-max aggregation has a false-positive rate that
    approximates the union of its members' FPRs (spec §1.2).
    """
    diffs = [abs(r.score - r.max_model_score)
             for r in results if r.model_scores and np.isfinite(r.score)]
    if not diffs:
        return False
    return float(np.mean(diffs)) <= tolerance
