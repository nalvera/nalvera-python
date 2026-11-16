"""Encrypted uploads (81-encryption.md): the NVLT v2 encryptor, the key pin,
the flow through a run, and "never fall back to plaintext"."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import os
import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import nvlt_reference
import stub_anonymizer
from conftest import scan_bytes
from nalvera import cli, encrypt
from nalvera.errors import EncryptionError
from nalvera.upload import Uploader

MODEL = "totalsegmentator_ct"
# The Privacy Vault's own encryptor, when its source sits beside this
# repository (or NALVERA_PRIVACY_VAULT_DIR names it); the cross-checks skip
# without it.
VAULT = Path(os.environ.get("NALVERA_PRIVACY_VAULT_DIR")
             or Path(__file__).resolve().parent.parent.parent / "Privacy-Vault")
SIDECAR = VAULT / "sidecar" / "vault_sidecar" / "crypto.py"


def _encrypt(data: bytes, public: bytes) -> bytes:
    out = io.BytesIO()
    encrypt.encrypt_stream(io.BytesIO(data), out, public)
    return out.getvalue()


# ── the container ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("size", [0, 1, 1000, (1 << 20) - 1, 1 << 20, (1 << 20) + 1,
                                  3 * (1 << 20) + 17])
def test_round_trip_against_an_independent_decryptor(size):
    private, public = nvlt_reference.new_key_pair()
    data = os.urandom(size)
    container = _encrypt(data, public)
    assert container[:4] == b"NVLT"
    assert nvlt_reference.decrypt(container, private) == data


def test_encryption_is_randomized():
    _private, public = nvlt_reference.new_key_pair()
    assert _encrypt(b"same scan", public) != _encrypt(b"same scan", public)


def test_a_wrong_key_cannot_read_it():
    _private, public = nvlt_reference.new_key_pair()
    other, _ = nvlt_reference.new_key_pair()
    with pytest.raises(nvlt_reference.NotDecryptable):
        nvlt_reference.decrypt(_encrypt(b"x" * 100, public), other)


def _frames(container: bytes):
    at, frames = 56, []
    while at < len(container):
        (length,) = struct.unpack(">I", container[at:at + 4])
        frames.append(container[at:at + 4 + length])
        at += 4 + length
    return container[:56], frames


def test_tampered_truncated_and_reordered_containers_are_rejected():
    private, public = nvlt_reference.new_key_pair()
    container = _encrypt(os.urandom(2 * (1 << 20) + 5), public)
    header, frames = _frames(container)
    assert len(frames) == 3
    flipped = bytearray(container)
    flipped[56 + 4 + 100] ^= 1
    header_edit = bytearray(container)
    header_edit[6] = 1                                     # the reserved field is authenticated
    cases = {
        "tampered": bytes(flipped),
        "header": bytes(header_edit),
        "truncated": header + frames[0] + frames[1],      # the last frame dropped
        "reordered": header + frames[1] + frames[0] + frames[2],
        "cut": container[:-10],
    }
    for name, broken in cases.items():
        with pytest.raises(nvlt_reference.NotDecryptable):
            nvlt_reference.decrypt(broken, private)
        assert name


def test_the_byte_layout_with_a_fixed_key_salt_and_ephemeral_key():
    recipient = X25519PrivateKey.from_private_bytes(bytes(range(32)))
    recipient_public = recipient.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ephemeral_raw, salt = bytes(range(100, 132)), bytes(range(200, 216))
    data = bytes(range(256)) * 4097                        # one full frame and a short one
    out = io.BytesIO()
    written = encrypt.encrypt_stream(io.BytesIO(data), out, recipient_public,
                                     _ephemeral=ephemeral_raw, _salt=salt)
    container = out.getvalue()
    assert written == len(container)
    # Header: magic, version 2, algorithm 1, reserved 0, ephemeral key, salt.
    ephemeral = X25519PrivateKey.from_private_bytes(ephemeral_raw)
    ephemeral_public = ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    header = b"NVLT" + bytes([2, 1, 0, 0]) + ephemeral_public + salt
    assert container[:56] == header
    # Frames, derived here from the format description alone.
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt,
               info=b"nalvera-vault/v2/x25519-aesgcm").derive(
        ephemeral.exchange(X25519PublicKey.from_public_bytes(recipient_public)))
    aead = AESGCM(key)
    first = aead.encrypt(b"\x00" * 4 + struct.pack(">Q", 0), data[:1 << 20], header + b"\x00")
    second = aead.encrypt(b"\x00" * 4 + struct.pack(">Q", 1), data[1 << 20:], header + b"\x01")
    expected = header + struct.pack(">I", len(first)) + first + \
        struct.pack(">I", len(second)) + second
    assert container == expected


def _sidecar():
    if not SIDECAR.exists():
        pytest.skip("the Privacy Vault sidecar is not at its path on this computer")
    spec = importlib.util.spec_from_file_location("vault_sidecar_crypto", SIDECAR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_privacy_vault_decrypts_what_the_package_encrypts():
    sidecar = _sidecar()
    private, public = nvlt_reference.new_key_pair()
    data = os.urandom((1 << 20) + 333)
    plain = io.BytesIO()
    sidecar.decrypt_stream(io.BytesIO(_encrypt(data, public)), plain, private)
    assert plain.getvalue() == data


def test_the_reference_decryptor_reads_what_the_privacy_vault_encrypts():
    sidecar = _sidecar()
    private, public = nvlt_reference.new_key_pair()
    data = os.urandom((1 << 20) + 333)
    out = io.BytesIO()
    sidecar.encrypt_stream(io.BytesIO(data), out, public)
    assert nvlt_reference.decrypt(out.getvalue(), private) == data


def test_the_trusted_fingerprint_is_the_privacy_vaults_pin():
    pin = Path(SIDECAR.parent / "recipient_key.py")
    if not pin.exists():
        pytest.skip("the Privacy Vault sidecar is not at its path on this computer")
    assert encrypt.TRUSTED_KEY_SHA256[0] in pin.read_text()


def test_encrypt_file_never_replaces_a_file_and_is_private(tmp_path):
    _private, public = nvlt_reference.new_key_pair()
    source = tmp_path / "a.nii.gz"
    source.write_bytes(b"scan")
    target = tmp_path / "a.nii.gz.nvlt"
    size = encrypt.encrypt_file(str(source), str(target), public)
    assert size == target.stat().st_size
    if os.name == "posix":
        assert target.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        encrypt.encrypt_file(str(source), str(target), public)


# ── the key pin ───────────────────────────────────────────────────────────────
def _block(*keys, available=True, required=False):
    return {"available": available, "required": required, "format": "NVLT/2",
            "keys": [{"public_key": base64.b64encode(k).decode(),
                      "sha256": hashlib.sha256(k).hexdigest()} for k in keys]}


def test_a_trusted_key_is_used_and_the_first_trusted_one_wins():
    _p1, untrusted = nvlt_reference.new_key_pair()
    _p2, trusted = nvlt_reference.new_key_pair()
    pins = (hashlib.sha256(trusted).hexdigest(),)
    assert encrypt.platform_key(_block(untrusted, trusted), "https://nalvera.ai",
                                trusted=pins) == trusted


def test_an_untrusted_key_is_never_used():
    _p, swapped = nvlt_reference.new_key_pair()
    with pytest.raises(EncryptionError) as caught:
        encrypt.platform_key(_block(swapped), "https://nalvera.ai")
    assert caught.value.code == "encrypt_no_trusted_key"


def test_a_local_server_is_trusted_without_a_pin():
    _p, key = nvlt_reference.new_key_pair()
    for url in ("http://localhost:8000", "http://127.0.0.1:5000"):
        assert encrypt.platform_key(_block(key), url) == key
    with pytest.raises(EncryptionError):
        encrypt.platform_key(_block(key), "https://localhost.example.org")


def test_a_platform_without_a_key():
    with pytest.raises(EncryptionError) as caught:
        encrypt.platform_key(_block(available=False), "https://nalvera.ai")
    assert caught.value.code == "encryption_unavailable"
    with pytest.raises(EncryptionError):
        encrypt.platform_key(None, "https://nalvera.ai")


def test_decide():
    _p, key = nvlt_reference.new_key_pair()
    local = "http://127.0.0.1:1"
    assert encrypt.decide(None, _block(key), local) is None
    assert encrypt.decide(False, _block(key), local) is None
    assert encrypt.decide(True, _block(key), local) == key
    assert encrypt.decide(None, _block(key, required=True), local) == key
    with pytest.raises(EncryptionError) as caught:
        encrypt.decide(False, _block(key, required=True), local)
    assert caught.value.code == "encrypt_required_by_account"


def test_without_the_cryptography_library_nothing_falls_back(monkeypatch):
    _p, key = nvlt_reference.new_key_pair()

    def missing():
        raise EncryptionError(code="encrypt_unavailable")

    monkeypatch.setattr(encrypt, "_crypto", missing)
    with pytest.raises(EncryptionError) as caught:
        encrypt.decide(True, _block(key), "http://127.0.0.1:1")
    assert caught.value.code == "encrypt_unavailable"
    assert encrypt.available() is False


# ── the flow through a run ────────────────────────────────────────────────────
def _uploads(server):
    return [u for u in server.uploads.values()]


def test_an_encrypted_run(make_client, server, scans, stub_anon, monkeypatch):
    server.enable_encryption()
    data = scan_bytes(b"patient", 2500)
    path = scans("Doe_John.nii.gz", data)
    seen = []
    real = encrypt.encrypt_file

    def spy(source, target, key):
        size = real(source, target, key)
        seen.append((source, target))
        return size

    monkeypatch.setattr(encrypt, "encrypt_file", spy)
    uploaded = []
    real_upload = Uploader.upload

    def spy_upload(self, upload_path, *, filename, size, display_name):
        # What exists at the moment the upload starts.
        uploaded.append({"plaintext_left": os.path.exists(seen[-1][0]), "path": upload_path,
                         "size": size, "on_disk": os.path.getsize(upload_path),
                         "filename": filename})
        return real_upload(self, upload_path, filename=filename, size=size,
                           display_name=display_name)

    monkeypatch.setattr(Uploader, "upload", spy_upload)
    task = make_client().run(path, model=MODEL, budget="personal", encrypt=True)[0]
    (up,) = _uploads(server)
    assert up["filename"] == f"{task.code}.nii.gz.nvlt"
    assert uploaded == [{"plaintext_left": False, "path": seen[0][1], "size": up["size"],
                         "on_disk": up["size"], "filename": up["filename"]}]
    assert bytes(up["data"])[:4] == b"NVLT" and up["size"] == len(up["data"])
    assert up["plaintext"] == data                         # the platform reads the scan
    assert server.tasks[task.id]["body"]["name"] == f"{task.code}.nii.gz"
    assert task.name == f"{task.code}.nii.gz"
    assert len(seen) == 1
    assert not any(os.path.exists(p) for pair in seen for p in pair)


def test_a_resumed_encrypted_upload_resends_the_same_container(make_client, server, scans,
                                                               stub_anon, monkeypatch):
    server.enable_encryption()
    data = scan_bytes(b"patient", 2500)
    path = scans("a.nii.gz", data)
    containers = []
    real = encrypt.encrypt_file

    def spy(source, target, key):
        size = real(source, target, key)
        containers.append(Path(target).read_bytes())
        return size

    monkeypatch.setattr(encrypt, "encrypt_file", spy)
    # The server stores a part, then the answer is lost.
    server.inject("PATCH", "/api/v1/uploads/", drop=True, after=True, times=1)
    make_client().run(path, model=MODEL, budget="personal", encrypt=True)
    (up,) = _uploads(server)
    assert len(containers) == 1 and bytes(up["data"]) == containers[0]
    assert up["plaintext"] == data
    assert server.calls("HEAD")


def test_the_account_switch_encrypts_without_being_asked(make_client, server, scans,
                                                         stub_anon):
    server.enable_encryption(required=True)
    make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    (up,) = _uploads(server)
    assert up["filename"].endswith(".nii.gz.nvlt") and up["plaintext"]


def test_the_encrypt_setting_from_the_environment(make_client, server, scans, stub_anon,
                                                  monkeypatch):
    server.enable_encryption()
    monkeypatch.setenv("NALVERA_ENCRYPT", "1")
    make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert _uploads(server)[0]["filename"].endswith(".nvlt")


@pytest.mark.parametrize("scene,code", [
    ("no_platform_key", "encryption_unavailable"),
    ("untrusted_key", "encrypt_no_trusted_key"),
    ("no_library", "encrypt_unavailable"),
    ("refused_by_account", "encrypt_required_by_account"),
    ("refused_by_account_env", "encrypt_required_by_account"),
])
def test_encryption_never_falls_back_to_plaintext(make_client, server, scans, stub_anon,
                                                  monkeypatch, scene, code):
    kwargs = {"encrypt": True}
    if scene == "untrusted_key":
        server.enable_encryption()
        monkeypatch.setattr(encrypt, "is_local", lambda url: False)
    elif scene == "no_library":
        server.enable_encryption()

        def missing():
            raise EncryptionError(code="encrypt_unavailable")

        monkeypatch.setattr(encrypt, "_crypto", missing)
    elif scene == "refused_by_account":
        server.enable_encryption(required=True)
        kwargs = {"encrypt": False}
    elif scene == "refused_by_account_env":
        server.enable_encryption(required=True)
        monkeypatch.setenv("NALVERA_ENCRYPT", "0")
        kwargs = {}
    with pytest.raises(EncryptionError) as caught:
        make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal", **kwargs)
    assert caught.value.code == code
    assert server.calls("POST", "/api/v1/uploads") == [] and stub_anonymizer.prepared == []


def test_the_server_refuses_a_plaintext_upload_when_the_account_requires_encryption(
        make_client, server, scans, stub_anon, monkeypatch):
    # An older client that does not know the switch: the server stops it at
    # the first bytes and the package says how to turn encryption on.
    server.enable_encryption(required=True)
    monkeypatch.setattr(encrypt, "decide", lambda *args, **kwargs: None)
    with pytest.raises(EncryptionError) as caught:
        make_client().run(scans("a.nii.gz"), model=MODEL, budget="personal")
    assert caught.value.code == "encryption_required" and caught.value.status == 422
    (up,) = _uploads(server)
    assert up["state"] == "cancelled" and len(up["data"]) == 0
    assert cli.exit_status(caught.value) == cli.EXIT_AUTH


def test_whoami_says_whether_uploads_are_encrypted(cli_dialect, env, server, monkeypatch,
                                                  no_sleep):
    from test_cli import run_cli
    assert cli.messages.text("whoami.encryption_unavailable") in run_cli("whoami")[1]
    server.enable_encryption()
    assert cli.messages.text("whoami.encryption_off") in run_cli("whoami")[1]
    monkeypatch.setenv("NALVERA_ENCRYPT", "1")
    assert cli.messages.text("whoami.encryption_on") in run_cli("whoami")[1]
    server.encryption_required = True
    assert cli.messages.text("whoami.encryption_required") in run_cli("whoami")[1]


def test_the_command_line_flags(env, server, stub_anon, monkeypatch, no_sleep):
    from test_cli import run_cli
    server.enable_encryption()
    scan = env / "a.nii.gz"
    scan.write_bytes(scan_bytes(b"a"))
    status, _out, err = run_cli("run", str(scan), "--model", MODEL, "--budget", "personal",
                                "--encrypt", "--no-wait")
    assert status == 0, err
    assert _uploads(server)[0]["filename"].endswith(".nvlt")
    server.encryption_required = True
    status, _out, _err = run_cli("run", str(scan), "--model", MODEL, "--budget", "personal",
                                 "--no-encrypt", "--no-wait", "--force")
    assert status == cli.EXIT_AUTH and len(server.uploads) == 1
    assert run_cli("run", str(scan), "--model", MODEL, "--encrypt", "--no-encrypt")[0] == 2
