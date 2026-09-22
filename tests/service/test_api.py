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
