"""The HTTP surface: six JSON endpoints and one page, over a real socket.

Tested through an actual server on a real port rather than by calling
handlers directly — routing, status codes and body framing are most of what
an HTTP layer can get wrong, and none of that is exercised by calling the
handler function.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from dfd.service.api import make_server
from dfd.service.store import Store
from dfd.service.worker import Worker


class _Record:
    verdict = "insufficient_evidence"
    llr_total = 0.0
    ood_score = 0.0

    def to_json(self) -> str:
        # Shaped like a real record rather than a one-key stub: the `rd`
        # projection reads `evidence`, `posterior` and `quality_band`, and a
        # stub without them would let a contract that drops all three pass.
        return json.dumps({
            "verdict": "insufficient_evidence",
            "llr_total": 0.0,
            "posterior": 0.5,
            "quality_band": "reject",
            "schema_version": "2",
            "evidence": [
                {"detector": "blend_seam", "llr": 0.0, "raw_score": None,
                 "abstained": True, "reason": "below_quality_floor"},
                {"detector": "gend_clip", "llr": 0.0, "raw_score": 0.266,
                 "abstained": True, "reason": "uncalibrated_for_band"},
            ],
        })


def _jpeg_bytes() -> bytes:
    """A small, genuinely decodable JPEG, built once for the upload tests."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(0)
    img = rng.integers(60, 190, (64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


_JPEG_BYTES = _jpeg_bytes()


@pytest.fixture()
def server(tmp_path: Path):
    store = Store(tmp_path / "db.sqlite3")
    worker = Worker(store=store, decide=lambda p: _Record(),
                    inbox=tmp_path / "inbox", workdir=tmp_path / "work")
    httpd = make_server("127.0.0.1", 0, store=store, worker=worker,
                        evidence_path=tmp_path / "evidence.json")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.base = f"http://127.0.0.1:{httpd.server_address[1]}"  # type: ignore[attr-defined]
    httpd.store = store  # type: ignore[attr-defined]
    httpd.worker = worker  # type: ignore[attr-defined]
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(server: Any, path: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(server.base + path, timeout=10) as r:
            body = r.read()
            ctype = r.headers.get("Content-Type", "")
            return r.status, (json.loads(body) if "json" in ctype else body)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, body


def _post(server: Any, path: str, data: bytes,
          ctype: str = "application/octet-stream",
          chunked: bool = False) -> tuple[int, Any]:
    if chunked:
        # urllib always sets Content-Length, so a chunked request has to be
        # framed by hand. Shared rather than inlined: two routes refuse
        # chunked and both must be shown to do it.
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1",
                                          server.server_address[1], timeout=10)
        conn.putrequest("POST", path)
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        conn.send(b"4\r\ntest\r\n0\r\n\r\n")
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        try:
            return resp.status, json.loads(raw)
        except ValueError:
            return resp.status, raw
    req = urllib.request.Request(server.base + path, data=data, method="POST",
                                 headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, body


def test_health_reports_ok_and_the_queue_depth(server: Any) -> None:
    status, body = _get(server, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["queue_depth"] == 0


def test_health_counts_a_queued_submission(server: Any) -> None:
    server.worker.submit_bytes(b"jpegbytes", filename="a.jpg")
    _, body = _get(server, "/health")
    assert body["queue_depth"] == 1


def test_posting_a_file_queues_it_and_returns_its_id(server: Any) -> None:
    status, body = _post(server, "/api/scan?filename=a.jpg", b"jpegbytes")
    assert status == 202
    assert server.store.get(body["id"]).status == "queued"


def test_a_posted_file_can_be_read_back_by_id(server: Any) -> None:
    _, body = _post(server, "/api/scan?filename=a.jpg", b"jpegbytes")
    server.worker.process_one()
    status, row = _get(server, f"/api/submissions/{body['id']}")
    assert status == 200
    assert row["status"] == "done"
    assert row["verdict"] == "insufficient_evidence"
    assert row["record"]["verdict"] == "insufficient_evidence"


def test_an_empty_post_is_refused_with_400_not_queued(server: Any) -> None:
    status, body = _post(server, "/api/scan?filename=a.jpg", b"")
    assert status == 400
    assert "empty" in body["error"]
    assert server.store.recent() == []


def test_an_upload_over_the_size_limit_is_refused(tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite3")
    worker = Worker(store=store, decide=lambda p: _Record(),
                    inbox=tmp_path / "inbox", workdir=tmp_path / "work")
    httpd = make_server("127.0.0.1", 0, store=store, worker=worker,
                        evidence_path=tmp_path / "e.json", max_upload_bytes=16)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, body = _post(httpd, "/api/scan?filename=a.jpg", b"x" * 17)
        assert status == 413
        assert store.recent() == []
    finally:
        httpd.shutdown(); httpd.server_close(); thread.join(timeout=5)


def test_an_unknown_submission_id_is_404_not_a_server_error(
        server: Any) -> None:
    status, body = _get(server, "/api/submissions/deadbeef")
    assert status == 404
    assert "error" in body


def test_the_listing_returns_newest_first_and_honours_limit(
        server: Any) -> None:
    for i in range(3):
        _post(server, f"/api/scan?filename={i}.jpg", b"x" * (i + 1))
    status, body = _get(server, "/api/submissions?limit=2")
    assert status == 200
    assert [r["filename"] for r in body["submissions"]] == ["2.jpg", "1.jpg"]


def test_the_listing_never_carries_the_full_audit_records(
        server: Any) -> None:
    """Fifty records of tens of kilobytes each is not a listing."""
    _post(server, "/api/scan?filename=a.jpg", b"jpegbytes")
    server.worker.process_one()
    _, body = _get(server, "/api/submissions")
    assert "record" not in body["submissions"][0]


def test_stats_are_served(server: Any) -> None:
    _post(server, "/api/scan?filename=a.jpg", b"jpegbytes")
    server.worker.process_one()
    status, body = _get(server, "/api/stats")
    assert status == 200
    assert body["status"]["done"] == 1


def test_the_dashboard_is_served_as_html(server: Any) -> None:
    status, body = _get(server, "/")
    assert status == 200
    assert b"<!doctype html>" in body.lower()


def test_the_evidence_endpoint_says_so_when_no_report_card_exists(
        server: Any) -> None:
    """A verdict with no measured detector performance behind it is a claim.

    The endpoint must not 404 into silence — the dashboard reads it to decide
    what warning to show.
    """
    status, body = _get(server, "/api/evidence")
    assert status == 200
    assert body["available"] is False


def test_the_evidence_endpoint_serves_the_report_card_when_present(
        tmp_path: Path) -> None:
    store = Store(tmp_path / "db.sqlite3")
    worker = Worker(store=store, decide=lambda p: _Record(),
                    inbox=tmp_path / "inbox", workdir=tmp_path / "work")
    card = tmp_path / "evidence.json"
    card.write_text(json.dumps({"detectors": {"blend_seam": {"auc": 0.289}}}))
    httpd = make_server("127.0.0.1", 0, store=store, worker=worker,
                        evidence_path=card)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, body = _get(httpd, "/api/evidence")
        assert status == 200
        assert body["available"] is True
        assert body["detectors"]["blend_seam"]["auc"] == 0.289
    finally:
        httpd.shutdown(); httpd.server_close(); thread.join(timeout=5)


def test_an_unknown_path_is_404(server: Any) -> None:
    status, _ = _get(server, "/nope")
    assert status == 404


def test_a_path_traversal_attempt_does_not_read_the_filesystem(
        server: Any) -> None:
    status, body = _get(server, "/../../etc/passwd")
    assert status == 404
    assert b"root:" not in (body if isinstance(body, bytes) else b"")


def test_a_negative_limit_does_not_return_the_whole_table(server):
    """SQLite reads `LIMIT -1` as NO limit.

    Verified 2026-09-23: `SELECT ... LIMIT ?` with -1 returns every row. So
    `?limit=-1` turned a bounded listing endpoint into an unbounded one, and
    the only visible symptom was a large response.
    """
    for i in range(5):
        server.worker.submit_bytes(b"x" * 10, filename=f"f{i}.png", source="api")

    status, body = _get(server, "/api/submissions?limit=-1")

    assert status == 400
    assert "at least 1" in body["error"]


def test_a_zero_limit_is_refused_rather_than_returning_nothing(server):
    """`LIMIT 0` returns no rows, which reads as 'no submissions' — a lie."""
    server.worker.submit_bytes(b"x" * 10, filename="f.png", source="api")

    status, body = _get(server, "/api/submissions?limit=0")

    assert status == 400
    assert "at least 1" in body["error"]


def test_a_non_integer_limit_answers_400_rather_than_closing_the_connection(server):
    """It used to raise inside the handler, which sends no response at all."""
    status, body = _get(server, "/api/submissions?limit=abc")

    assert status == 400
    assert "must be an integer" in body["error"]


def test_a_large_limit_is_clamped_to_the_ceiling(server):
    """A caller asking for more than the ceiling gets the ceiling, not an error.

    Asserted on the parser rather than over the wire: proving the clamp
    through HTTP would need 500+ submissions, and a test that inserts one row
    and asks for 100,000 passes whether or not any clamp exists.
    """
    from dfd.service.api import MAX_LISTING_LIMIT, _listing_limit

    assert _listing_limit("100000") == MAX_LISTING_LIMIT
    assert _listing_limit(str(MAX_LISTING_LIMIT + 1)) == MAX_LISTING_LIMIT
    assert _listing_limit("7") == 7

    server.worker.submit_bytes(b"x" * 10, filename="f.png", source="api")
    status, body = _get(server, "/api/submissions?limit=100000")
    assert status == 200
    assert len(body["submissions"]) == 1


def test_a_limit_actually_bounds_the_listing(server):
    for i in range(4):
        server.worker.submit_bytes(b"x" * 10, filename=f"f{i}.png", source="api")

    status, body = _get(server, "/api/submissions?limit=2")

    assert status == 200
    assert len(body["submissions"]) == 2


def test_a_chunked_upload_is_refused_rather_than_stored_empty(server):
    """This handler reads Content-Length bytes and cannot decode chunked.

    Without the check the body arrives as length 0 and is accepted as an
    EMPTY submission, which then fails at decode — recording a refusal about
    the file rather than about the request.
    """
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                      timeout=10)
    conn.putrequest("POST", "/api/scan?filename=chunky.png")
    conn.putheader("Transfer-Encoding", "chunked")
    conn.endheaders()
    conn.send(b"4\r\ntest\r\n0\r\n\r\n")
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()

    assert resp.status == 411
    assert "chunked" in body["error"]
    # And nothing was queued: an accepted-but-empty submission is the defect.
    assert server.store.recent(limit=10) == []


def test_a_truncated_body_is_refused_rather_than_stored_short(server):
    """A caller that promises 100 bytes and sends 10 must not be stored as 10."""
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                      timeout=10)
    conn.putrequest("POST", "/api/scan?filename=short.png")
    conn.putheader("Content-Length", "100")
    conn.endheaders()
    conn.send(b"0123456789")
    conn.sock.shutdown(1)  # half-close: no more body is coming
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()

    assert resp.status == 400
    assert "of 100 bytes" in body["error"]
    assert server.store.recent(limit=10) == []


def test_the_handler_bounds_how_long_a_connection_may_stall(server):
    """Slow-read denial of service: without a timeout one stalled caller holds
    a worker thread until the process dies."""
    from dfd.service.api import DEFAULT_REQUEST_TIMEOUT_S, _Handler

    assert _Handler.timeout == DEFAULT_REQUEST_TIMEOUT_S
    assert 0 < _Handler.timeout <= 120


def test_concurrent_requests_are_capped_rather_than_unbounded(tmp_path):
    """ThreadingHTTPServer spawns a thread per connection with no ceiling."""
    from dfd.service.api import make_server

    store = Store(tmp_path / "db.sqlite3")
    worker = Worker(store=store, decide=lambda p: _Record(),
                    inbox=tmp_path / "inbox", workdir=tmp_path / "work")
    httpd = make_server("127.0.0.1", 0, store=store, worker=worker,
                        evidence_path=tmp_path / "evidence.json",
                        max_concurrent=1)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        # Take the only slot and hold it, then ask for another.
        httpd.request_slots.acquire()
        req = urllib.request.Request(base + "/health")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 503
        httpd.request_slots.release()
        # And every served request RETURNS its slot. Two successive requests,
        # not one: with a single slot and no release, the first still
        # succeeds on the slot just handed back and only the second exposes
        # the leak.
        for _ in range(3):
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                assert r.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_rd_format_is_served_for_a_scored_submission(server: Any) -> None:
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg", source="api")
    server.worker.process_one()
    status, body = _get(server, f"/api/submissions/{sid}?format=rd")
    assert status == 200
    assert set(body) >= {"verdict", "score", "models", "dissenting_models"}
    assert body["verdict"] in {"AUTHENTIC", "MANIPULATED", "UNKNOWN"}


def test_rd_format_keeps_the_abstentions_visible(server: Any) -> None:
    # The point of the contract on THIS deployment: every detector is gated
    # out of deciding, so the abstentions and their reasons are the answer.
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg", source="api")
    server.worker.process_one()
    _, body = _get(server, f"/api/submissions/{sid}?format=rd")
    assert {m["name"] for m in body["models"]} == {"blend_seam", "gend_clip"}
    assert all(m["abstained"] for m in body["models"])
    assert body["dissenting_models"] == []


def test_the_plain_format_is_unchanged_by_the_new_parameter(server: Any) -> None:
    # Everything already reading /api/submissions/<id> must keep seeing the
    # audit record. A format parameter that quietly reshapes the default
    # response is a breaking change wearing an opt-in's clothes.
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg", source="api")
    server.worker.process_one()
    status, body = _get(server, f"/api/submissions/{sid}")
    assert status == 200 and "record" in body and "models" not in body


def test_an_unknown_format_is_refused_rather_than_silently_ignored(
        server: Any) -> None:
    # Ignoring it would answer `?format=rd2` with the audit record, which a
    # caller would read as "this build serves rd2".
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg", source="api")
    server.worker.process_one()
    status, body = _get(server, f"/api/submissions/{sid}?format=nope")
    assert status == 400
    assert "format" in body["error"]


def test_rd_format_on_an_unscored_submission_says_so_rather_than_guessing(
        server: Any) -> None:
    # A queued submission has no verdict. Returning UNKNOWN would be
    # indistinguishable from "scored, and nothing could be concluded", which
    # is a completely different statement about the same image.
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg", source="api")
    status, body = _get(server, f"/api/submissions/{sid}?format=rd")
    assert status == 409
    assert body["status"] == "queued"


def test_rd_format_on_a_FAILED_submission_does_not_say_come_back_later(
        server: Any) -> None:
    # `Store.fail` writes `error` and leaves record_json NULL, so a branch
    # testing record_json alone answered 409 "not scored yet" -- for ever --
    # to a client whose image can never be scored. The synchronous
    # /api/analyze path has always answered 422 with the reason; these now
    # agree rather than contradicting each other about the same row.
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg", source="api")
    # fail() only moves a RUNNING row, so claim it first — the same
    # transition the worker makes before it tries to decode.
    assert server.worker.store.claim_next() is not None
    server.worker.store.fail(sid, "decode failed: not an image")
    status, body = _get(server, f"/api/submissions/{sid}?format=rd")
    assert status == 422
    assert body["status"] == "failed"
    assert "decode failed" in body["error"]


def test_rd_format_on_an_unknown_id_is_still_a_404(server: Any) -> None:
    status, _ = _get(server, "/api/submissions/nope?format=rd")
    assert status == 404


def test_analyze_scores_synchronously_and_returns_the_contract(
        server: Any) -> None:
    status, body = _post(server, "/api/analyze?filename=f.jpg", _JPEG_BYTES)
    assert status == 200
    assert set(body) >= {"verdict", "score", "models", "dissenting_models"}


def test_analyze_leaves_the_same_audit_trail_as_scan(server: Any) -> None:
    # Synchronous must not mean unrecorded. A decision a client received and
    # the service cannot show you afterwards is the one kind this project
    # cannot have.
    _post(server, "/api/analyze?filename=f.jpg", _JPEG_BYTES)
    rows = server.store.recent(limit=10)
    assert len(rows) == 1 and rows[0].status == "done"


def test_analyze_enforces_the_same_upload_limit_as_scan(tmp_path: Path) -> None:
    # A dedicated 16-byte-limit server, exactly as the /api/scan size test
    # does: the refusal happens from the Content-Length header before the
    # body is read, so posting a default-sized over-limit body would have
    # the server close mid-write and the client see a broken pipe instead
    # of the 413 it actually sent.
    store = Store(tmp_path / "db.sqlite3")
    worker = Worker(store=store, decide=lambda p: _Record(),
                    inbox=tmp_path / "inbox", workdir=tmp_path / "work")
    httpd = make_server("127.0.0.1", 0, store=store, worker=worker,
                        evidence_path=tmp_path / "e.json", max_upload_bytes=16)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, _ = _post(httpd, "/api/analyze?filename=f.jpg", b"x" * 17)
        assert status == 413
        assert store.recent() == []
    finally:
        httpd.shutdown(); httpd.server_close(); thread.join(timeout=5)


def test_analyze_refuses_chunked_like_scan_does(server: Any) -> None:
    # Reused, not re-derived: an /api/analyze that reads a chunked body as
    # length 0 accepts an EMPTY submission and reports a decode failure about
    # content the caller never finished sending.
    status, _ = _post(server, "/api/analyze", b"", chunked=True)
    assert status == 411


def test_analyze_refuses_an_empty_upload(server: Any) -> None:
    status, _ = _post(server, "/api/analyze?filename=f.jpg", b"")
    assert status == 400


def test_the_dashboard_escapes_every_value_it_interpolates(server: Any) -> None:
    """A filename is chosen by whoever submits, and the dashboard renders it.

    `POST /api/scan?filename=...` keeps `<`, `>` and quotes -- `api._post`
    applies only `Path(...).name` -- and a file dropped in the watched inbox
    reaches the same cell. Unescaped, one submission runs script in every
    operator's dashboard, same-origin, with access to the full audit records
    at /api/submissions/<id> and to /api/scan itself.
    """
    _, raw = _get(server, "/")
    page = raw.decode("utf-8")
    # The helper exists, and no row cell interpolates a server value raw.
    assert "const esc = (v) =>" in page
    for raw in ("${r.filename}", "${r.source}", "${r.status}",
                "${d.corpus", "${d.note", "${n}</code>"):
        assert raw not in page, f"unescaped interpolation still present: {raw}"


def test_a_scripted_filename_survives_round_trip_as_data(server: Any) -> None:
    # The payload reaches the store intact (it is a legal filename), which
    # is exactly why rendering has to escape rather than rely on input
    # filtering.
    evil = '<img src=x onerror=alert(1)>.jpg'
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename=evil, source="api")
    status, body = _get(server, f"/api/submissions/{sid}")
    assert status == 200
    assert body["filename"] == evil


def test_analyze_serves_the_simple_format(server: Any) -> None:
    status, body = _post(server, "/api/analyze?format=simple", _JPEG_BYTES)
    assert status == 200
    assert set(body) >= {"is_deepfake", "confidence", "reason",
                         "why_no_answer", "detectors_consulted"}
    # On this deployment nothing is measured above the evidence floor, so
    # the honest answer is null -- and NEVER 0.5, which a caller could
    # threshold back into a verdict.
    assert body["is_deepfake"] is None
    assert body["confidence"] is None
    assert body["reason"] == "insufficient_evidence"


def test_the_two_formats_never_disagree_about_one_submission(
        server: Any) -> None:
    # `simple` is a projection of the same record, not a second opinion.
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg",
                                     source="api")
    assert server.worker.process_one()
    _, rd = _get(server, f"/api/submissions/{sid}?format=rd")
    _, simple = _get(server, f"/api/submissions/{sid}?format=simple")
    assert simple["verdict"] == rd["verdict"]
    assert simple["detectors_consulted"] == len(rd["models"])
    if rd["verdict"] == "UNKNOWN":
        assert simple["is_deepfake"] is None


def test_an_unknown_format_is_refused_on_both_routes(server: Any) -> None:
    # Answering `?format=rd2` with something would read as "this build
    # serves rd2", and a caller would ship against a format that is not here.
    status, body = _get(server, "/api/submissions/x?format=rd2")
    assert status in (400, 404)
    status, body = _post(server, "/api/analyze?format=rd2", _JPEG_BYTES)
    assert status == 400
    assert "simple" in body["error"] and "rd" in body["error"]


def test_analyze_works_with_no_filename_at_all(server: Any) -> None:
    """`?filename=` is optional: the bytes name themselves.

    Dispatch is by extension, so a caller posting bytes and no name used to
    get "unsupported file extension ''" — a problem they did not cause and
    could not act on.
    """
    status, body = _post(server, "/api/analyze?format=simple", _JPEG_BYTES)
    assert status == 200
    assert "is_deepfake" in body
    assert body["reason"] == "insufficient_evidence"


def test_a_supplied_extension_is_never_overridden_by_the_bytes(
        server: Any) -> None:
    # Dispatch-by-extension is the contract the rest of the package is
    # tested against; sniffing only fills in a name that was missing.
    status, body = _post(server, "/api/scan?filename=photo.png", _JPEG_BYTES)
    assert status == 202
    row = server.worker.store.get(body["id"])
    assert row.filename == "photo.png"


def test_an_unrecognisable_stream_is_left_unnamed(server: Any) -> None:
    """Guessing wrong is worse than not guessing.

    A wrong suffix routes the file to a decoder that fails confusingly, or
    silently reads one frame of a video as an image. Leaving the name alone
    lets `dfd.pipeline`'s existing, clear extension error fire instead --
    which it does on the real ingest path; this `server` fixture stubs
    `decide`, so what is checked HERE is the API's own job: whether it
    completed the name.
    """
    status, body = _post(server, "/api/scan", b"this is just prose" * 20)
    assert status == 202
    row = server.worker.store.get(body["id"])
    assert row.filename == "upload", "an unrecognised stream was given a suffix"


def test_a_name_without_an_extension_is_completed_from_the_bytes(
        server: Any) -> None:
    status, body = _post(server, "/api/scan?filename=holiday", _JPEG_BYTES)
    assert status == 202
    row = server.worker.store.get(body["id"])
    assert row.filename == "holiday.jpg"
