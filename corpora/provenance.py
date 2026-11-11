"""What a report says about where its corpus came from, without saying where it lives.

WHY A REPORT MAY NOT CARRY A FILESYSTEM PATH. This repository is PUBLIC, and every bench and
training report in it is committed. `str(args.captures)` is the obvious thing to write into a
provenance field and it published `/home/<operator>/Desktop/vkyc-server/captures` -- the
operator's home directory layout, and the location of a corpus of biometric v-CIP recordings
whose data-protection position is still open (docs/HANDOFF.md, owner-action item 6). No capture
data ever leaked; the path did, in six tracked files, and the fix is that no writer can emit one.

WHY THIS DOES NOT TRY TO NAME THE CORPUS. An earlier version of this module mapped directory
names onto `assets/manifest.yaml` ids, so `.../captures` was reported as `vkyc_capture_corpus`.
That is a FALSE PROVENANCE CLAIM in a field whose only job is honest provenance: any directory
called `captures` anywhere would have been stamped as the biometric corpus, and a pytest
tmp_path fixture promptly was. Worse, it erased a distinction the manifest says matters -- two
different v-CIP capture directories exist and they are not the same corpus. The bare directory
name keeps them apart and claims nothing that was not checked by a human. Asserting that a run
used a particular manifest id is a human's claim and belongs in the manifest or the evidence
card, which is where the gate already reads it from
(`dfd.service.evidence.gated_detectors`).
"""
from __future__ import annotations

from pathlib import Path


def corpus_label(path: str | Path | None) -> str:
    """The corpus directory's own name. NEVER a path, never an inferred id.

    The return value is one path component at most.
    `tests/corpora/test_provenance.py` asserts that as an invariant over
    adversarial inputs rather than trusting any table to be complete.
    """
    if path is None:
        return "unknown"
    # Split on BOTH separators. `Path` on POSIX does not treat "\\" as one,
    # so a Windows path would otherwise pass through whole and the field
    # would carry exactly the thing this function exists to remove. Caught
    # by the invariant test, not by review.
    text = str(path).replace("\\", "/")
    parts = [p for p in text.split("/") if p not in ("", ".", "..")]
    return parts[-1] if parts else "unknown"
