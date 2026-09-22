"""The evidence gate: measured performance decides who may decide.

A detector in this repo can be in one of three states, and only the third may
influence a verdict:

1. **No weights.** It abstains with `weights_absent`. Visible, harmless.
2. **Weights, but not measured above the floor.** It still runs and its raw
   score is still written into the audit record — the data is worth having —
   but it gets no calibration curve, so `Calibrator.to_evidence` returns
   llr 0.0 with `uncalibrated_for_band` and it cannot move the fused result.
3. **Measured above the floor.** It is calibrated and it decides.

State 2 is the one this module exists for. Every measured-but-useless
detector in the history of this field shipped because "it is better than
nothing" was decided by the person who built it rather than by a number, and
nothing in a codebase stops that unless something fails closed.

Today no detector on this deployment is in state 3 — see
`bench/evidence_card.json` and `docs/HANDOFF.md §0`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

#: Minimum measured AUC, on a corpus the detector did not train on, before a
#: detector is allowed to contribute evidence to a verdict. 0.75 is not a
#: rounded-up version of what this project's detectors score; it is a floor
#: below which the false-positive cost at a fraud base rate makes the output
#: unusable for the identity-verification decision this is meant to support.
#: Raise it per deployment; the card cannot lower it (see `gated_detectors`).
DEFAULT_AUC_FLOOR = 0.75

CARD_FORMAT_VERSION = 1

#: The committed card, relative to the repository root.
EVIDENCE_CARD_PATH = Path(__file__).resolve().parents[3] / "bench" / "evidence_card.json"


class CardError(ValueError):
    """The evidence card is missing, unreadable, or of an unknown format.

    Its own exception type because the caller must not treat it as "nothing
    measured": an absent card and a card full of failures both end with no
    detector deciding, and only this distinguishes a misconfigured deployment
    from an honest one.
    """


def load_card(path: str | Path = EVIDENCE_CARD_PATH) -> dict[str, Any]:
    """Read the measured-performance card.

    Raises:
        CardError: if the file is absent, malformed, or declares a format
            version this code does not implement.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError as exc:
        raise CardError(f"no evidence card at {p}") from exc
    except (OSError, ValueError) as exc:
        raise CardError(f"unreadable evidence card {p}: {exc}") from exc
    if data.get("format_version") != CARD_FORMAT_VERSION:
        raise CardError(
            f"evidence card {p} declares format_version "
            f"{data.get('format_version')!r}; this code implements "
            f"{CARD_FORMAT_VERSION}")
    if not isinstance(data.get("detectors"), dict):
        raise CardError(f"evidence card {p} has no detectors mapping")
    return cast("dict[str, Any]", data)


def gated_detectors(card: dict[str, Any],
                    floor: float = DEFAULT_AUC_FLOOR) -> set[str]:
    """Names allowed to contribute evidence, by measured AUC.

    Args:
        card: as returned by `load_card`.
        floor: minimum measured AUC. The card's own `auc_floor` is used only
            when it is STRICTER than this one — a card cannot weaken the
            caller's bar, or the gate becomes a field the thing being gated
            gets to fill in.

    Returns:
        The set of detector names at or above the effective floor. A detector
        with `auc: null` (never measured) is excluded: unmeasured fails
        closed, exactly as an unregistered asset does in the manifest.
    """
    effective = max(float(floor), float(card.get("auc_floor", floor)))
    allowed: set[str] = set()
    for name, entry in card["detectors"].items():
        auc = entry.get("auc")
        if auc is None:
            logger.info("detector %s: never measured, cannot decide", name)
            continue
        if float(auc) < effective:
            logger.info("detector %s: measured AUC %.3f below the %.2f floor, "
                        "cannot decide", name, float(auc), effective)
            continue
        allowed.add(name)
    return allowed
