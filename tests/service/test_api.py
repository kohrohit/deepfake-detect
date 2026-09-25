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
        return json.dumps({"verdict": "insufficient_evidence"})


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
          ctype: str = "application/octet-stream") -> tuple[int, Any]:
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
