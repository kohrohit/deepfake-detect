"""Shared secure weight-loading machinery for torch-based detectors (spec §3A).

Extracted verbatim out of npr.py: this ~150-line block is security-critical
and hardened over four review rounds (supply-chain gate, cache staleness,
cache-key collisions across factories, actionable error messages). Every
weights-parameterised detector (NPR, the EffNet-backed slots A/E, ...) needs
the exact same machinery. Copying it would mean a future security fix has to
land twice — and the second copy is the one that gets forgotten. So it lives
here once, and detectors import and call `load_model`.

Supply-chain control: weights are loaded with torch.load(..., weights_only=True)
by default, preventing arbitrary code execution from tampered models. Full-module
pickles must be explicitly enabled via allow_unsafe_load and are logged as a
warning every time.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Module-level cache for loaded weights, keyed by load configuration (not just
# path) so a frozen detector dataclass need not carry mutable model state, and
# so distinct (factory, allow_unsafe_load) configurations over the same file
# never collide. Guarded by a lock against concurrent first-load races.
_MODEL_CACHE: dict[tuple[str, int, int, str, bool], nn.Module] = {}
_CACHE_LOCK = threading.Lock()


def load_model(
    weights_path: str | Path,
    model_factory: Callable[[], nn.Module] | None = None,
    allow_unsafe_load: bool = False,
) -> nn.Module:
    """Load a torch model from disk securely, with caching and staleness detection.

    Three paths, in order of preference:

    1. **Secure path (recommended):** model_factory is provided.
       Load state_dict via weights_only=True and instantiate model via
       model_factory. No warnings. This is the secure-by-default path.

    2. **Rejection path:** model_factory is None and file is a full module.
       Reject with a clear error message directing the caller to path 1 or 3.

    3. **Unsafe path (last resort):** allow_unsafe_load=True.
       Load full module via unpickle (arbitrary code execution possible).
       Logs a WARNING every time it is taken.

    Note: a state_dict alone cannot become a model without knowing the
    architecture. Secure loading therefore requires the architecture to be
    known (via model_factory). This is a genuine constraint, not a limitation
    of this implementation.

    Uses a module-level cache keyed by (resolved_path, mtime_ns, size,
    factory_id, allow_unsafe_load) to detect file replacement and to
    distinguish different load configurations. The same file loaded with
    different factories or security modes can produce different models and
    must not share a cache entry. Protected by a lock against concurrent
    first-load races.

    Args:
        weights_path: path to the weights file.
        model_factory: callable that returns an uninitialized model instance
            (e.g. `lambda: torch.nn.Linear(3, 2)`). If provided, enables the
            secure weights_only=True path and the weights file must be a
            state_dict. If None, only full-module pickles with
            allow_unsafe_load=True are accepted.
        allow_unsafe_load: if True, permits loading full-module pickles via
            unpickling (arbitrary code execution). Logs a WARNING when taken.

    Returns:
        the loaded, `.eval()`'d model as an nn.Module.

    Raises:
        RuntimeError: if the file is a full-module pickle and model_factory
            is None and allow_unsafe_load is False (names model_factory as
            the remedy); or if the file is a full-module pickle AND
            model_factory is provided (names the two as mutually exclusive);
            or if weights_only load succeeds but yields a state_dict while
            model_factory is None.
        ValueError: if a loaded object is not a state_dict when model_factory
            is provided, or not an nn.Module when loading a full module.
        Exception: any other exception from torch.load propagates.
    """
    resolved = Path(weights_path).resolve()
    stat = resolved.stat()

    # Compute a stable identity for the factory so two different factories
    # over the same file never share a cache entry.
    if model_factory is None:
        factory_id = "none"
    else:
        factory_id = (
            f"{model_factory.__module__}."
            f"{getattr(model_factory, '__qualname__', repr(model_factory))}"
        )

    cache_key = (
        str(resolved),
        stat.st_mtime_ns,
        stat.st_size,
        factory_id,
        allow_unsafe_load,
    )

    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            logger.debug("using cached model from %s", resolved)
            return _MODEL_CACHE[cache_key]

        logger.debug("loading model from %s", resolved)

        # CASE 1: Secure path — model_factory provided
        if model_factory is not None:
            try:
                sd = torch.load(str(resolved), map_location="cpu", weights_only=True)
                if not isinstance(sd, dict):
                    raise ValueError(
                        f"expected state_dict (dict) with model_factory, "
                        f"got {type(sd).__name__}"
                    )
                model = model_factory()
                model.load_state_dict(sd)
                model.eval()
                logger.debug("loaded state_dict securely with model_factory")
                _MODEL_CACHE[cache_key] = model
                return model
            except Exception as e:
                # Provide a clear error if the file appears to be a full-module pickle
                if isinstance(e, Exception) and "Weights only load failed" in str(e):
                    raise RuntimeError(
                        f"the file {resolved} appears to be a full-module pickle, "
                        f"not a state_dict. model_factory requires a state_dict. "
                        f"These are mutually exclusive: either (1) provide a state_dict "
                        f"file with model_factory, or (2) remove model_factory and use "
                        f"allow_unsafe_load=True for full pickles."
                    ) from e
                logger.error(
                    "failed to load state_dict with model_factory: %s",
                    type(e).__name__,
                    exc_info=True,
                )
                raise

        # CASE 2: Attempt weights_only=True (expecting a full module saved safely)
        try:
            model = torch.load(str(resolved), map_location="cpu", weights_only=True)
            # weights_only succeeded but model_factory is None
            if isinstance(model, dict):
                raise RuntimeError(
                    f"loaded a state_dict but model_factory is None. "
                    f"To securely load weights, provide model_factory as "
                    f"a callable that returns an uninitialized model instance."
                )
            if not isinstance(model, nn.Module):
                raise ValueError(
                    f"expected torch.nn.Module, got {type(model).__name__}"
                )
            model.eval()
            logger.debug("loaded full module with weights_only=True")
            _MODEL_CACHE[cache_key] = model
            return model
        except Exception as e:
            # weights_only failed (likely a full module pickle)
            if not allow_unsafe_load:
                raise RuntimeError(
                    f"failed to load {resolved} with weights_only=True. "
                    f"This is a supply-chain security measure (spec §3A). "
                    f"To fix: (1) provide model_factory to load state_dict, or "
                    f"(2) set allow_unsafe_load=True (logs warning). "
                    f"Error: {type(e).__name__}"
                ) from e

            # CASE 3: Unsafe path — operator has explicitly opted in
            logger.warning(
                "loading %s with full unpickle (allow_unsafe_load=True); "
                "this permits arbitrary code execution from the weights file",
                resolved,
            )
            model = torch.load(str(resolved), map_location="cpu")
            if not isinstance(model, nn.Module):
                raise ValueError(
                    f"unsafe load: expected torch.nn.Module, "
                    f"got {type(model).__name__}"
                )
            model.eval()
            logger.debug("loaded full module with unsafe unpickle")
            _MODEL_CACHE[cache_key] = model
            return model
