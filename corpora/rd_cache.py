"""Loader for cached Reality Defender responses (spec §1.2).

These are the free head-to-head data: 24 results with per-model breakdowns,
already paid for. No quota is consumed by reading them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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
    skipped: list[str] = []
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
            # Validate and coerce every field here, inside the try. Nothing
            # past this point should be able to raise TypeError/ValueError/
            # KeyError for reasons unrelated to bad input — those exceptions
            # must still mean "malformed result", not "bug in our own
            # field-building code" or "bug in the constructor call below",
            # which is why RDResult(...) is built outside this block.
            cache_key = path.parent.name
            verdict = d.get("verdict", "")
            score = float(d.get("score", float("nan")))
        except (json.JSONDecodeError, OSError, TypeError, ValueError,
                KeyError) as exc:
            # Skip a result that parses but is the wrong shape (a list, a
            # string, a non-numeric score, a model entry missing a field)
            # exactly as a decode/IO failure is skipped — one bad cache entry
            # must not take down the other 23. A dropped RD result is a
            # dropped row in the head-to-head comparison, so this is never
            # silent either.
            logger.warning("skipping unreadable RD result %s: %s",
                           path.parent.name, exc)
            skipped.append(path.parent.name)
            continue
        # Outside the try: a typo'd keyword or missing required field here
        # is a defect in this module, not bad input, and must raise.
        out.append(RDResult(
            cache_key=cache_key,
            verdict=verdict,
            score=score,
            model_scores=model_scores,
            model_verdicts=model_verdicts,
        ))
    if skipped:
        logger.warning("loaded %d RD cache results, skipped %d: %s",
                       len(out), len(skipped), ", ".join(skipped))
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
