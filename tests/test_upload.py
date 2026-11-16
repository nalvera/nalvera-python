"""Resumable uploads against the fake server."""
from __future__ import annotations

import base64
import hashlib
import threading

import pytest

from conftest import scan_bytes
from fake_server import error_body
from nalvera import _http, auth
from nalvera.errors import NalveraError, ScanRejected, TooLarge
from nalvera.upload import Uploader


@pytest.fixture
def transport(env, server, fake_time):
    creds = auth.Credentials(email=server.email, api_key=auth.ApiKey(server.key))
    t = _http.Transport(server.url, creds, clock=fake_time.clock, sleep=fake_time.sleep,
                        rand=lambda: 0.5)
    yield t
    t.close()


@pytest.fixture
def said():
    return []


@pytest.fixture
def uploader(transport, said):
    return Uploader(transport, min_poll_seconds=5,
                    say=lambda key, /, **fields: said.append((key, fields)))


@pytest.fixture
def scan_file(tmp_path):
    data = scan_bytes(b"patient", 2500)
    path = tmp_path / "scan-AAAAAAAA.nii.gz"
    path.write_bytes(data)
    return path, data


def _the_upload(server):
    assert len(server.uploads) == 1
    return next(iter(server.uploads.values()))


def test_parts_with_checksums_and_completion_in_the_last_part(uploader, server, scan_file):
    path, data = scan_file
    result = uploader.upload(str(path), filename=path.name, size=len(data),
                             display_name="Doe_John_CT.nii.gz")
    up = _the_upload(server)
    assert bytes(up["data"]) == data and up["state"] == "ready"
    assert result.upload_id == up["id"] and result.receipt["shape"] == [4, 4, 4]
    patches = server.calls("PATCH")
    assert len(patches) == 3                                   # 1000-byte parts
    offsets = [int(p["headers"]["Upload-Offset"]) for p in patches]
    assert offsets == [0, 1000, 2000]
    for p, start in zip(patches, offsets):
        part = data[start:start + 1000]
        expected = base64.b64encode(hashlib.sha256(part).digest()).decode()
        assert p["headers"]["Upload-Checksum"] == f"sha256 {expected}"
        assert p["headers"]["Content-Type"] == "application/offset+octet-stream"
    assert "Upload-Complete" not in patches[0]["headers"]
    assert patches[-1]["headers"]["Upload-Complete"] == "?1"
    assert server.calls("POST", "/complete") == []             # completed by the last part
    created = server.calls("POST", "/api/v1/uploads")[0]
    assert created["body_length"] < 200                        # name and size only


def test_progress_names_the_original_file_only(uploader, said, scan_file, server):
    path, data = scan_file
    uploader.upload(str(path), filename=path.name, size=len(data), display_name="Doe_John.nii.gz")
    assert said and all(fields.get("name") == "Doe_John.nii.gz" for _k, fields in said)
    assert said[-1][0] == "upload.done"
    names = [r["headers"].get("X-Nalvera-Email") for r in server.calls()]
    assert all("Doe_John" not in str(n) for n in names)


def test_a_lost_answer_resumes_from_the_servers_offset(uploader, server, scan_file):
    path, data = scan_file
    # The server stores the second part, then the answer is lost.
    server.inject("PATCH", "/api/v1/uploads/", drop=True, after=True, times=1)
    uploader.upload(str(path), filename=path.name, size=len(data), display_name="x")
    up = _the_upload(server)
    assert bytes(up["data"]) == data                          # nothing sent twice
    assert server.calls("HEAD")


def test_a_connection_dropped_before_the_server_reads_resends_the_part(uploader, server,
                                                                       scan_file):
    path, data = scan_file
    server.inject("PATCH", "/api/v1/uploads/", drop=True, times=2)
    uploader.upload(str(path), filename=path.name, size=len(data), display_name="x")
    assert bytes(_the_upload(server)["data"]) == data
    assert len(server.calls("HEAD")) == 2


def test_offset_mismatch_continues_where_the_server_says(uploader, server, scan_file):
    path, data = scan_file
    server.inject("PATCH", "/api/v1/uploads/", status=409, times=1,
                  body=error_body("offset_mismatch", offset=0))
    uploader.upload(str(path), filename=path.name, size=len(data), display_name="x")
    assert bytes(_the_upload(server)["data"]) == data


def test_a_damaged_part_is_sent_again(uploader, server, scan_file):
    path, data = scan_file
    server.inject("PATCH", "/api/v1/uploads/", status=460, times=2,
                  body=error_body("checksum_mismatch", offset=0))
    uploader.upload(str(path), filename=path.name, size=len(data), display_name="x")
    assert bytes(_the_upload(server)["data"]) == data


def test_parts_damaged_again_and_again_end_the_upload(uploader, server, scan_file):
    path, data = scan_file
    server.inject("PATCH", "/api/v1/uploads/", status=460, times=100,
                  body=error_body("checksum_mismatch", offset=0))
    with pytest.raises(NalveraError) as caught:
        uploader.upload(str(path), filename=path.name, size=len(data), display_name="x")
    assert caught.value.code == "upload_failed"
    assert server.calls("DELETE")                               # the server's slot is freed


def test_an_answer_ahead_of_the_stored_bytes_is_corrected_by_the_server(transport, server,
                                                                        scan_file):
    path, data = scan_file
    up = Uploader(transport, min_poll_seconds=5)
    # An answer acknowledges a part the server did not keep: the next part
    # meets offset_mismatch and the upload continues from the true offset.
    server.inject("PATCH", "/api/v1/uploads/", status=200, body={"offset": 1000}, times=1)
    up.upload(str(path), filename=path.name, size=len(data), display_name="x")
    assert bytes(_the_upload(server)["data"]) == data


def test_a_file_that_is_not_gzip_is_rejected(uploader, server, tmp_path):
    path = tmp_path / "scan-BBBBBBBB.nii.gz"
    path.write_bytes(b"PLAINTEXT" * 300)
    with pytest.raises(ScanRejected) as caught:
        uploader.upload(str(path), filename=path.name, size=path.stat().st_size,
                        display_name="x")
    assert caught.value.code == "scan_rejected" and caught.value.extra["reason"]


def test_a_scan_the_server_rejects_after_checking(uploader, server, tmp_path):
    path = tmp_path / "scan-CCCCCCCC.nii.gz"
    path.write_bytes(scan_bytes(b"mask", 1500) + b"REJECT")
    with pytest.raises(ScanRejected) as caught:
        uploader.upload(str(path), filename=path.name, size=path.stat().st_size,
                        display_name="x")
    assert caught.value.extra["reason"]


def test_checking_by_a_separate_service_is_polled_politely(uploader, server, scan_file,
                                                           fake_time):
    path, data = scan_file
    server.complete_mode = "service"
    before = fake_time.slept
    uploader.upload(str(path), filename=path.name, size=len(data), display_name="x")
    polls = server.calls("GET", "/api/v1/uploads/")
    assert len(polls) == 2                                      # processing, then ready
    waited = fake_time.slept - before
    assert waited >= 2.0 + 5.0                                  # server hint, then the minimum


def test_too_many_uploads_is_waited_out(uploader, server, scan_file, fake_time):
    path, data = scan_file
    server.inject("POST", "/api/v1/uploads", status=429, times=2, headers={"Retry-After": "3"},
                  body=error_body("too_many_uploads", retry_after=3, limit=4))
    uploader.upload(str(path), filename=path.name, size=len(data), display_name="x")
    assert len(server.calls("POST", "/api/v1/uploads")) == 3
    assert fake_time.slept >= 6


def test_a_file_over_the_limit_is_refused_before_any_request(transport, server, scan_file):
    path, data = scan_file
    up = Uploader(transport, max_file_bytes=100)
    with pytest.raises(TooLarge):
        up.upload(str(path), filename=path.name, size=len(data), display_name="x")
    assert server.calls() == []


def test_at_most_two_files_upload_at_a_time(transport, server, tmp_path):
    server.patch_delay = 0.05
    up = Uploader(transport, min_poll_seconds=5)
    files = []
    for i in range(5):
        p = tmp_path / f"scan-{i}AAAAAAA.nii.gz"
        p.write_bytes(scan_bytes(str(i).encode(), 2500))
        files.append(p)
    errors = []

    def go(p):
        try:
            up.upload(str(p), filename=p.name, size=p.stat().st_size, display_name=p.name)
        except Exception as exc:                                 # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(p,)) for p in files]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert server.max_active_patches <= 2
    assert all(u["state"] == "ready" for u in server.uploads.values())
