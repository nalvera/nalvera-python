"""Resumable uploads of anonymized scans.

The server's protocol, as the website uses it:

1. ``POST /api/v1/uploads`` ``{"filename", "size"}`` starts an upload and
   names the part size (``chunk_bytes``).
2. ``PATCH /api/v1/uploads/{id}`` sends one part at ``Upload-Offset`` with
   an ``Upload-Checksum`` (SHA-256), so a part damaged on the way is sent
   again. The last part carries ``Upload-Complete: ?1``.
3. When an answer is lost, ``HEAD`` gives the offset the server stored, and
   the upload continues from there. A part sent at the wrong offset is
   answered with the right one (``offset_mismatch``).
4. The server checks the scan; ``GET`` reports ``ready`` or ``rejected``.

At most two files upload at a time. ``too_many_uploads`` (the account's
uploads in progress) is waited out by the transport.
"""
from __future__ import annotations

import base64
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from . import messages
from ._http import Cancelled, Transport
from .errors import (
    RETRYABLE_CODES,
    NalveraError,
    NetworkError,
    ScanRejected,
    ServiceUnavailable,
    TooLarge,
    UploadInterrupted,
    bounded,
)

OCTET_STREAM = "application/offset+octet-stream"
DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
#: The largest part size and check interval taken from an answer.
MAX_CHUNK_BYTES = 256 * 1024 * 1024
MAX_POLL_AFTER_MS = 300_000.0
MAX_PARALLEL_UPLOADS = 2
SETTLED_STATES = ("ready", "rejected", "failed", "cancelled", "expired", "consumed")
_MB = 1024 * 1024

Say = Callable[..., None]


@dataclass
class UploadResult:
    """A scan the server stored and checked: ``upload_id`` is what a task
    request names; ``receipt`` holds the server's check (``shape``...)."""
    upload_id: str
    size: int
    receipt: Dict[str, Any] = field(default_factory=dict)


class Uploader:
    """Uploads files through a ``Transport``.

    ``say(key, **fields)`` prints a progress text (a messages key). Setting
    the transport's ``stop`` event ends uploads between parts
    (``Cancelled``).
    """

    #: The refusals a part or the completion waits out and sends again. Beside
    #: the "try again" codes: ``encryption_unavailable``, the platform's
    #: decryption key missing for a while (an operator's fault); the part was
    #: not stored and the upload stays resumable.
    RETRY_CODES = RETRYABLE_CODES | {"encryption_unavailable"}

    def __init__(self, transport: Transport, *, min_poll_seconds: float = 5.0,
                 max_file_bytes: Optional[int] = None, say: Optional[Say] = None,
                 max_part_failures: int = 5, max_resumes: int = 20,
                 max_check_seconds: float = 1800.0):
        self.t = transport
        self.min_poll_seconds = max(1.0, float(min_poll_seconds))
        self.max_file_bytes = max_file_bytes
        self.say = say or (lambda key, /, **fields: None)
        self.max_part_failures = max_part_failures
        self.max_resumes = max_resumes
        self.max_check_seconds = max_check_seconds
        self._slots = threading.BoundedSemaphore(MAX_PARALLEL_UPLOADS)

    # ── public ────────────────────────────────────────────────────────────────
    def upload(self, path: str, *, filename: str, size: int, display_name: str) -> UploadResult:
        """Upload the file at ``path`` as ``filename`` (the neutral code
        name) and wait until the server has checked it. ``display_name`` (the
        original name) is used in progress texts only, never sent.

        Raises ``ScanRejected`` when the server refuses the scan, and
        ``NalveraError`` (``upload_failed``) when the upload cannot
        finish.
        """
        if self.max_file_bytes and size > self.max_file_bytes:
            raise TooLarge(status=None, extra={"max_bytes": self.max_file_bytes})
        with self._slots:
            self._check_cancel()
            created = self.t.json("POST", "/api/v1/uploads", idempotent=True,
                                  json={"filename": filename, "size": int(size)})
            upload_id = str(created["upload_id"])
            # The part size the server names, within sense.
            chunk = int(bounded(created.get("chunk_bytes"), 1, MAX_CHUNK_BYTES,
                                DEFAULT_CHUNK_BYTES) or DEFAULT_CHUNK_BYTES)
            try:
                status = self._transfer(upload_id, path, int(size), chunk, display_name)
                status = self._until_checked(upload_id, status, display_name)
            except BaseException:
                self._abandon(upload_id)
                raise
        return UploadResult(upload_id=upload_id, size=int(size),
                            receipt=dict(status.get("receipt") or {}))

    # ── sending ───────────────────────────────────────────────────────────────
    def _transfer(self, upload_id: str, path: str, size: int, chunk: int,
                  name: str) -> Dict[str, Any]:
        """Send every byte and complete the upload. Returns the first status
        object after completion."""
        offset = 0
        # Two uploads run at once: what one of them knows stays in its state.
        state = {"part_failures": 0, "resumes": 0, "reported": 0, "size": size}
        with open(path, "rb") as fh:
            while True:
                status, offset = self._send_parts(upload_id, fh, offset, size, chunk, name, state)
                if status is not None:
                    return status
                try:
                    # The server may check the scan within this request.
                    response = self.t.request("POST", f"/api/v1/uploads/{upload_id}/complete",
                                              idempotent=True, timeout=(10.0, 300.0),
                                              retry_codes=self.RETRY_CODES)
                except UploadInterrupted as missing:
                    # Bytes are missing on the server: continue where it says.
                    offset = self._resume_point(missing, offset, state)
                    continue
                return _body(response)

    def _send_parts(self, upload_id: str, fh, offset: int, size: int, chunk: int, name: str,
                    state: Dict[str, int]):
        """Send parts from ``offset`` to the end. Returns (status, offset):
        status is the status object when the last part completed the upload,
        else None (the caller completes it)."""
        while offset < size:
            self._check_cancel()
            fh.seek(offset)
            part = fh.read(min(chunk, size - offset))
            last = offset + len(part) >= size
            headers = {
                "Upload-Offset": str(offset),
                "Content-Type": OCTET_STREAM,
                "Upload-Checksum": "sha256 " + base64.b64encode(
                    hashlib.sha256(part).digest()).decode("ascii"),
            }
            if last:
                headers["Upload-Complete"] = "?1"
            try:
                response = self.t.request("PATCH", f"/api/v1/uploads/{upload_id}", data=part,
                                          headers=headers, idempotent=False,
                                          timeout=(10.0, 300.0), retry_codes=self.RETRY_CODES)
            except UploadInterrupted as refused:
                if refused.code == "checksum_mismatch":
                    state["part_failures"] += 1
                    if state["part_failures"] > self.max_part_failures:
                        raise self._failed(name, messages.text("error.upload_gave_up")) from None
                if refused.extra.get("busy"):
                    # An earlier request of ours still holds the part.
                    self.t.pause(self.min_poll_seconds)
                offset = self._resume_point(refused, offset, state)
                continue
            except (NetworkError, ServiceUnavailable) as lost:
                # The answer may be lost after the server stored the part:
                # ask where it stands and continue from there.
                state["resumes"] += 1
                if state["resumes"] > self.max_resumes:
                    raise lost
                self.t.pause(min(30.0, 2.0 ** min(state["resumes"], 5)))
                head = self._head(upload_id, size)
                if head["state"] != "receiving":
                    return self.t.json("GET", f"/api/v1/uploads/{upload_id}"), size
                offset = head["offset"]
                self.say("upload.resume", name=name, offset_mb=_mb(offset))
                continue
            body = _body(response)
            if "state" in body:
                self._report(name, size, size, state)
                return body, size
            offset = _offset(body.get("offset", offset + len(part)), size, offset)
            self._report(name, offset, size, state)
        return None, offset

    def _resume_point(self, refused: UploadInterrupted, offset: int,
                      state: Dict[str, int]) -> int:
        state["resumes"] += 1
        if state["resumes"] > self.max_resumes:
            raise refused
        return _offset(refused.extra.get("offset"), state["size"], offset)

    def _head(self, upload_id: str, size: int) -> Dict[str, Any]:
        response = self.t.request("HEAD", f"/api/v1/uploads/{upload_id}")
        return {"offset": _offset(response.headers.get("Upload-Offset"), size, 0),
                "state": response.headers.get("Upload-State", "receiving")}

    # ── the server's check ────────────────────────────────────────────────────
    def _until_checked(self, upload_id: str, status: Dict[str, Any], name: str) -> Dict[str, Any]:
        first, waited = True, 0.0
        while status.get("state") not in SETTLED_STATES:
            self._check_cancel()
            hint_s = (bounded(status.get("poll_after_ms"), 0.0, MAX_POLL_AFTER_MS, 0.0)
                      or 0.0) / 1000.0
            # Right after completing, the server's hint; afterwards never
            # faster than the account's minimum poll interval.
            delay = hint_s if first and hint_s else max(self.min_poll_seconds, hint_s)
            if first:
                self.say("run.checking", name=name)
            first = False
            if waited + delay > self.max_check_seconds:
                raise self._failed(name, messages.text("error.temporarily_unavailable"))
            self.t.pause(delay)
            waited += delay
            status = self.t.json("GET", f"/api/v1/uploads/{upload_id}")
        if status["state"] == "ready":
            return status
        receipt = status.get("receipt")
        reason = (receipt.get("error") if isinstance(receipt, dict) else None) \
            or status.get("error")
        if status["state"] == "rejected":
            raise ScanRejected(status=422, detail=reason, extra={"reason": reason or ""})
        raise self._failed(name, reason or status["state"])

    # ── helpers ───────────────────────────────────────────────────────────────
    def _report(self, name: str, done: int, total: int, state: Dict[str, int]) -> None:
        if total <= 0:
            return
        if done >= total:
            if state["reported"] < 100:
                state["reported"] = 100
                self.say("upload.done", name=name)
            return
        percent = int(done * 100 / total)
        milestone = percent - percent % 25
        if milestone > state["reported"] and milestone > 0:
            state["reported"] = milestone
            self.say("upload.progress", name=name, percent=milestone,
                     done_mb=_mb(done), total_mb=_mb(total))

    def _check_cancel(self) -> None:
        if self.t.stop.is_set():
            raise Cancelled()

    def _abandon(self, upload_id: str) -> None:
        """Best effort: free the server's upload slot."""
        try:
            self.t.request("DELETE", f"/api/v1/uploads/{upload_id}", max_total_wait=0,
                           timeout=(5.0, 10.0))
        except Exception:
            pass

    @staticmethod
    def _failed(name: str, reason: str) -> NalveraError:
        return NalveraError(messages.text("error.upload_failed", name=name, reason=reason),
                            code="upload_failed")


def _body(response) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _mb(n: int) -> str:
    return f"{n / _MB:.1f}"


def _offset(value, size: int, fallback: int) -> int:
    """An upload offset from an answer, between 0 and the file's size;
    ``fallback`` when the answer names none."""
    number = bounded(value, 0, max(0, int(size)))
    return int(number) if number is not None else int(fallback)
