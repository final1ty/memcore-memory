
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
import os, base64

class AES256GCM:
    def __init__(self, key: bytes):
        assert len(key) == 32
        self.aesgcm = AESGCM(key)

    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        kdf = Argon2id(salt=salt, length=32, iterations=3, lanes=4, memory_cost=64*1024)
        return kdf.derive(password.encode())

    @staticmethod
    def generate_key() -> bytes:
        return AESGCM.generate_key(bit_length=256)

    def encrypt(self, plaintext: bytes, associated_data: bytes = b""):
        nonce = os.urandom(12)
        ct = self.aesgcm.encrypt(nonce, plaintext, associated_data or None)
        return nonce, ct

    def decrypt(self, nonce: bytes, ciphertext: bytes, associated_data: bytes = b"") -> bytes:
        return self.aesgcm.decrypt(nonce, ciphertext, associated_data or None)

    def encrypt_str(self, s: str) -> str:
        nonce, ct = self.encrypt(s.encode())
        return f"{base64.b64encode(nonce).decode()}.{base64.b64encode(ct).decode()}"

    def decrypt_str(self, token: str) -> str:
        n_b64, ct_b64 = token.split(".")
        import base64 as b64
        nonce = b64.b64decode(n_b64)
        ct = b64.b64decode(ct_b64)
        return self.decrypt(nonce, ct).decode()
