"""Errors: one class and code per server code, instructions, no key in any text."""
from __future__ import annotations

import pytest

import nalvera
from fake_server import make_key
from nalvera import errors, messages

# Every code the platform sends, with its status and class.
SERVER_CODES = [
    (400, "invalid_request", errors.InvalidRequest),
    (400, "https_required", errors.ConfigError),
    (400, "key_in_url", errors.AuthenticationError),
    (401, "invalid_credentials", errors.AuthenticationError),
    (401, "key_expired", errors.AuthenticationError),
    (401, "key_revoked", errors.AuthenticationError),
    (402, "insufficient_credits", errors.InsufficientCredits),
    (403, "api_disabled", errors.ApiDisabled),
    (403, "api_blocked", errors.ApiDisabled),
    (403, "email_unverified", errors.PermissionDenied),
    (403, "terms_reaccept_required", errors.PermissionDenied),
    (403, "scope_missing", errors.PermissionDenied),
    (403, "ip_not_allowed", errors.PermissionDenied),
    (403, "lab_not_allowed", errors.PermissionDenied),
    (404, "not_found", errors.NotFound),
    (409, "not_ready", errors.NotReady),
    (409, "not_cancellable", errors.NotCancellable),
    (409, "offset_mismatch", errors.UploadInterrupted),
    (460, "checksum_mismatch", errors.UploadInterrupted),
    (413, "too_large", errors.TooLarge),
    (422, "scan_rejected", errors.ScanRejected),
    (426, "client_too_old", errors.ClientTooOld),
    (429, "rate_limited", errors.RateLimited),
    (429, "too_many_active_tasks", errors.RateLimited),
    (429, "too_many_uploads", errors.RateLimited),
    (500, "server_error", errors.ServerError),
    (503, "api_unavailable", errors.ServiceUnavailable),
    (503, "queue_full", errors.ServiceUnavailable),
    (503, "storage_full", errors.ServiceUnavailable),
    (503, "temporarily_unavailable", errors.ServiceUnavailable),
]


@pytest.mark.parametrize("status,code,cls", SERVER_CODES)
def test_each_server_code_has_its_class_and_an_instruction(status, code, cls):
    error = errors.from_response(status, {"detail": "server words", "code": code})
    assert isinstance(error, cls) and isinstance(error, nalvera.NalveraError)
    assert (error.code, error.status) == (code, status)
    assert messages.has(f"error.{code}")
    assert str(error)


def test_only_the_try_again_codes_are_retryable():
    assert errors.RETRYABLE_CODES == {"rate_limited", "too_many_active_tasks",
                                      "too_many_uploads", "queue_full", "storage_full",
                                      "temporarily_unavailable"}


def test_extra_fields_fill_the_instruction():
    error = errors.from_response(403, {"code": "scope_missing", "detail": "x",
                                       "required_scope": "results:read"})
    assert error.required_scope == "results:read"
    assert str(error) == messages.text("error.scope_missing", required_scope="results:read")
    credits = errors.from_response(402, {"code": "insufficient_credits", "budget": "lab:12"})
    assert str(credits) == messages.text("error.insufficient_credits", budget="lab:12")


def test_retry_after_is_read_from_the_body():
    error = errors.from_response(429, {"code": "rate_limited", "retry_after": 30})
    assert error.retry_after == 30.0


def test_an_unknown_code_degrades_to_the_servers_detail():
    error = errors.from_response(409, {"code": "something_new", "detail": "Plain words."})
    assert type(error) is errors.NalveraError and str(error) == "Plain words."
    assert error.code == "something_new"


def test_an_answer_without_a_code_gets_one_from_its_status():
    assert errors.from_response(502, None).code == "temporarily_unavailable"
    assert errors.from_response(500, "oops").code == "server_error"
    assert errors.from_response(404, {}).code == "not_found"


def test_a_key_in_any_text_is_redacted():
    key = make_key()
    error = errors.from_response(400, {"code": "invalid_request", "detail": f"got {key} here"})
    assert key not in str(error) and key not in (error.detail or "")
    assert key[:12] in str(error)
    assert errors.redact(f"a {key} b {key}") == f"a {key[:12]}…[redacted] b {key[:12]}…[redacted]"
    assert key not in errors.NalveraError(f"message with {key}").message


def test_budget_choice_required_lists_its_options():
    error = errors.BudgetChoiceRequired([
        {"budget": "personal", "name": "Personal", "available_credits": 3.0},
        {"budget": "lab:12", "name": "Imaging Lab", "available_credits": 99.5}])
    assert error.code == "budget_choice_required"
    assert "lab:12" in str(error) and "personal" in str(error)
    assert "99.50" in str(error)


def test_every_error_class_is_exported():
    for name in ("NalveraError", "NoCredentials", "AuthenticationError", "PermissionDenied",
                 "ApiDisabled", "InsufficientCredits", "BudgetChoiceRequired", "ScanRejected",
                 "RateLimited", "NotFound", "ClientTooOld"):
        assert issubclass(getattr(nalvera, name), nalvera.NalveraError)
