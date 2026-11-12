"""A provenance field may never carry a filesystem path."""
from __future__ import annotations

import pytest

from corpora.provenance import corpus_label


@pytest.mark.parametrize("path", [
    "/home/rohit/Desktop/vkyc-server/captures",
    "/home/ubuntu/datasets/vkyc_captures",
    "/home/ubuntu/datasets/fairface_sessions",
    "/Users/someone/secret/place/df40_eval",
    "relative/path/sfhq_part3",
    "/a/b/c/an_unknown_corpus",
    "/",
    "",
    None,
    "trailing/slash/captures/",
    "C:\\Users\\someone\\captures",
])
def test_no_label_can_carry_a_path(path):
    # The invariant, asserted over the table being INCOMPLETE: an unknown
    # corpus must still come back as one component, because the table is
    # what gets forgotten when a new corpus is added, and the leak must not
    # depend on somebody remembering.
    label = corpus_label(path)
    assert "/" not in label
    assert "\\" not in label
    assert not label.startswith(("home", "Users", "."))
    assert "rohit" not in label and "ubuntu" not in label


def test_the_label_is_the_directory_name_and_claims_nothing_more():
    # Deliberately NOT "vkyc_capture_corpus". Inferring a manifest id from a
    # directory name stamped every directory called `captures` -- a pytest
    # tmp_path fixture included -- as the biometric corpus, which is a false
    # claim in the one field that exists to be true.
    assert corpus_label("/home/op/Desktop/vkyc-server/captures") == "captures"
    assert corpus_label("/home/op/datasets/vkyc_captures") == "vkyc_captures"
    assert corpus_label("/home/op/datasets/fairface_sessions") == "fairface_sessions"


def test_two_different_capture_directories_stay_distinguishable():
    # The manifest records that two v-CIP capture directories exist and are
    # NOT the same corpus (42 sessions / 142 frames versus 146 / 554).
    # Collapsing both onto one id would have destroyed that distinction in
    # every report that carries it.
    assert corpus_label("/a/captures") != corpus_label("/b/vkyc_captures")
