
from pathlib import Path
from .aes_gcm import AES256GCM
import os, sys, json, getpass
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

MASTER_PASSWORD_ENV = "MNEM_MASTER_PASSWORD"

class MasterPasswordRequired(RuntimeError):
    """Raised when a password-protected key must be unlocked but no password is available."""

class KeyManager:
    def __init__(self, key_path: Path):
        self.key_path = key_path
        self._key = None

    def _resolve_password(self, password: str = None) -> str:
        """Explicit argument, then MNEM_MASTER_PASSWORD, then an interactive prompt.

        Only prompts when stdin is a real terminal - under a stdio MCP server, systemd
        or any other non-interactive parent, getpass() would consume the protocol stream
        (or abort), so raise a diagnosable error instead.
        """
        if password:
            return password
        env_password = os.environ.get(MASTER_PASSWORD_ENV)
        if env_password:
            return env_password
        try:
            interactive = sys.stdin is not None and sys.stdin.isatty()
        except (AttributeError, ValueError):
            interactive = False
        if interactive:
            return getpass.getpass("Mnemosyne master password: ")
        raise MasterPasswordRequired(
            f"{self.key_path} is password-protected but no password was supplied and stdin "
            f"is not a terminal. Pass --password, or set {MASTER_PASSWORD_ENV}."
        )

    def load_or_create(self, password: str = None) -> bytes:
        if self.key_path.exists() and self.key_path.stat().st_size > 100:
            data = json.loads(self.key_path.read_text())
            salt = bytes.fromhex(data['salt'])
            password = self._resolve_password(password)
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
