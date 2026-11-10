"""Embed once, measure many times.

WHY THIS EXISTS. One ViT-L/14 forward pass is ~81 GFLOPs. Every number spec §7 demands is a
REFIT, not a single fit: a permutation null is 12 of them, a grouped bootstrap is 1000 resamples,
a per-fold-refit LOGO is 6 folds. Re-running the encoder for each would put the measurement out of
reach of the CPU-only box this project deploys on. Cached, each of those is a logistic regression
over a matrix already in memory, and the whole validation suite costs ONE pass over the corpus.

WHERE THE CACHE LIVES. Outside the repository -- `~/Desktop/agents/datasets/clip_cache/`, beside
every other corpus here. A few hundred thousand `.npy` files inside a git worktree is a mistake
nobody undoes, and `git status` becomes unusable long before anyone notices.

WHAT ADDRESSES AN ENTRY. The pixels, their shape, the preprocessing version and the encoder id --
never a file path. Paths lie: this project's own capture corpus holds 979 byte-identical copies of
one demo asset, which cost one forward pass here instead of 979. Content addressing is also what
makes the cache safe: there is no way to serve one image's embedding for another's, because the
only thing that can produce a key is the image itself.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch.nn as nn

from dfd.detectors.clip_vit import PREPROCESS_VERSION
from dfd.detectors.gend import embed

logger = logging.getLogger(__name__)


class FeatureCache:
    """Content-addressed embeddings on disk, one `.npy` per image.

    `stats` counts hits and misses for the life of the instance and is meant
    to be logged by every caller: a run that reports 0 hits on a second pass
    has a key that is not stable, and that is worth noticing before it costs
    an afternoon of forward passes.

    A miss is an INPUT not served from disk, which is not the same as a
    forward pass: `embed_all` deduplicates by key first, so a call holding
    979 copies of one image reports 979 misses and runs the encoder once.
    """

    def __init__(self, root: str | Path, *, encoder_id: str) -> None:
        self.root = Path(root)
        self.encoder_id = encoder_id
        self.stats: dict[str, int] = {"hits": 0, "misses": 0}

    def key(self, image: npt.NDArray[np.uint8]) -> str:
        """The address of this image's embedding.

        Reads `PREPROCESS_VERSION` from the module at call time rather than
        capturing it at construction, so a change to resize or normalisation
        invalidates every entry rather than silently serving vectors computed
        under the old rule.
        """
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(image).tobytes())
        h.update(repr(image.shape).encode("utf-8"))
        h.update(PREPROCESS_VERSION.encode("utf-8"))
        h.update(self.encoder_id.encode("utf-8"))
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        # Two-character fan-out: 256 directories, so a corpus of 200k
        # embeddings does not put 200k entries in one of them.
        return self.root / key[:2] / f"{key}.npy"

    def get(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32] | None:
        """The stored embedding, or None. Never raises on a damaged entry."""
        path = self._path(self.key(image))
        if not path.exists():
            return None
        try:
            return np.load(path).astype(np.float32, copy=False)
        except Exception:
            # A half-written file from an interrupted run reads as a miss and
            # is recomputed. Raising here would make every resumed run fail
            # on the one entry the interruption caught.
            logger.warning("unreadable cache entry %s; recomputing", path,
                           exc_info=True)
            return None

    def put(self, image: npt.NDArray[np.uint8],
            vector: npt.NDArray[np.float32]) -> None:
        """Store one embedding, creating its directory if needed."""
        path = self._path(self.key(image))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(vector, dtype=np.float32))

    def embed_all(self, tower: nn.Module,
                  images: Sequence[npt.NDArray[np.uint8]],
                  *, batch_size: int = 8) -> npt.NDArray[np.float32]:
        """`(N, EMBED_DIM)` in input order, computing only what is missing.

        Each miss is written as it is computed rather than at the end, so an
        interrupted run resumes instead of starting over -- which, on a
        corpus this is worth using at all, is the difference between losing
        minutes and losing the afternoon.
        """
        out: list[npt.NDArray[np.float32] | None] = []
        # Misses grouped by key, NOT a flat list of indices. Duplicates
        # inside one call are the common case here -- the capture corpus
        # holds 979 byte-identical copies of one asset -- and embedding each
        # separately would both cost 979 forward passes and return 979 rows
        # that disagree in their last bits, because a batched matmul
        # reassociates differently depending on where in the batch a row
        # sits. One forward pass per distinct image, copied to every
        # position that asked for it.
        pending: dict[str, list[int]] = {}
        for i, image in enumerate(images):
            cached = self.get(image)
            if cached is None:
                self.stats["misses"] += 1
                pending.setdefault(self.key(image), []).append(i)
                out.append(None)
            else:
                self.stats["hits"] += 1
                out.append(cached)

        representatives = [positions[0] for positions in pending.values()]
        for start in range(0, len(representatives), batch_size):
            chunk = representatives[start:start + batch_size]
            vectors = embed(tower, [images[i] for i in chunk]).cpu().numpy()
            for i, vector in zip(chunk, vectors, strict=True):
                row = np.asarray(vector, dtype=np.float32)
                self.put(images[i], row)
                for position in pending[self.key(images[i])]:
                    out[position] = row

        if not out:
            return np.empty((0, 0), dtype=np.float32)
        if any(row is None for row in out):  # pragma: no cover - unreachable
            # Every None is filled by the loop above. Asserted rather than
            # filtered because filtering would return FEWER rows than inputs
            # and every caller zips this against a label vector -- the
            # resulting misalignment scores perfectly and means nothing.
            raise AssertionError(
                f"{sum(r is None for r in out)} of {len(out)} embeddings were "
                f"not computed; refusing to return a matrix shorter than its "
                f"input")
        return np.stack([row for row in out if row is not None])
