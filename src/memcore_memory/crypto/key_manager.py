
from pathlib import Path
from .aes_gcm import AES256GCM
import os, json, getpass
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

class KeyManager:
    def __init__(self, key_path: Path):
        self.key_path = key_path
        self._key = None

    def load_or_create(self, password: str = None) -> bytes:
        if self.key_path.exists() and self.key_path.stat().st_size > 100:
            data = json.loads(self.key_path.read_text())
            salt = bytes.fromhex(data['salt'])
            if not password:
                password = getpass.getpass("Mnemosyne master password: ")
            kdf = Argon2id(salt=salt, length=32, iterations=3, lanes=4, memory_cost=64*1024)
            kek = kdf.derive(password.encode())
            aes = AES256GCM(kek)
            key = aes.decrypt(bytes.fromhex(data['nonce']), bytes.fromhex(data['ct']))
            self._key = key
            return key
        else:
            master_key = AES256GCM.generate_key()
            if password:
                salt = os.urandom(16)
                kdf = Argon2id(salt=salt, length=32, iterations=3, lanes=4, memory_cost=64*1024)
                kek = kdf.derive(password.encode())
                aes = AES256GCM(kek)
                nonce, ct = aes.encrypt(master_key)
                self.key_path.write_text(json.dumps({'salt': salt.hex(),'nonce': nonce.hex(),'ct': ct.hex()}))
            else:
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                self.key_path.write_bytes(master_key)
            self._key = master_key
            return master_key

    @property
    def key(self) -> bytes:
        if not self._key:
            raise RuntimeError("Key not loaded")
        return self._key

    def get_cipher(self) -> AES256GCM:
        return AES256GCM(self.key)
