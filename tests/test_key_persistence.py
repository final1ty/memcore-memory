"""The master key must survive being loaded.

This is a regression test for the worst bug in the project's history. ``load_or_create``
branched on ``key_path.stat().st_size > 100``, so a raw 32-byte unprotected key - which
is what every store created without a password has - never matched, fell through to the
*create* branch, and got overwritten with a freshly generated key on every single open.

It was invisible in normal use: the process that created the store held the right key in
memory and kept serving correctly. Only the *next* process to open the store found the
data undecryptable, by which point the original key was gone for good. It fired for real
on the SkyNAS container on 2026-09-22.
"""

import json
import pytest

from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager, MasterKeyUnreadable


def test_unprotected_key_is_stable_across_loads(tmp_path):
    path = tmp_path / "master.key"
    first = KeyManager(path).load_or_create()
    on_disk = path.read_bytes()
    second = KeyManager(path).load_or_create()
    third = KeyManager(path).load_or_create()
    assert first == second == third
    assert path.read_bytes() == on_disk, "the key file was rewritten by a plain load"


def test_data_encrypted_before_a_reload_still_decrypts(tmp_path):
    """The actual consequence: ciphertext outliving its key."""
    path = tmp_path / "master.key"
    key = KeyManager(path).load_or_create()
    nonce, ct = AES256GCM(key).encrypt(b"SkyNAS Docker-only telepites sikeres")

    reopened = KeyManager(path).load_or_create()
    assert AES256GCM(reopened).decrypt(nonce, ct) == b"SkyNAS Docker-only telepites sikeres"


def test_password_protected_key_is_stable_across_loads(tmp_path):
    path = tmp_path / "master.key"
    first = KeyManager(path).load_or_create(password="hunter2")
    second = KeyManager(path).load_or_create(password="hunter2")
    assert first == second
    assert json.loads(path.read_text()).keys() >= {"salt", "nonce", "ct"}


def test_raw_key_file_is_not_mistaken_for_a_wrapped_one(tmp_path):
    path = tmp_path / "master.key"
    key = KeyManager(path).load_or_create()
    assert len(path.read_bytes()) == 32
    # No password anywhere: a raw key must load without one, not prompt or raise.
    assert KeyManager(path).load_or_create() == key


def test_unrecognisable_key_file_is_refused_rather_than_replaced(tmp_path):
    path = tmp_path / "master.key"
    path.write_bytes(b"not a key, not json, and not 32 bytes long either")
    with pytest.raises(MasterKeyUnreadable):
        KeyManager(path).load_or_create()
    assert path.read_bytes().startswith(b"not a key"), "the unreadable file was clobbered"


def test_new_key_file_is_not_world_readable(tmp_path):
    path = tmp_path / "master.key"
    KeyManager(path).load_or_create()
    assert path.stat().st_mode & 0o077 == 0
