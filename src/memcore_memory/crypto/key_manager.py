
from pathlib import Path
from .aes_gcm import AES256GCM
import os, sys, json, getpass
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

MASTER_PASSWORD_ENV = "MNEM_MASTER_PASSWORD"
# Deployment marker, deliberately not a MNEM_ setting: it gates whether an
# unprotected key may be created at all, which is a property of the environment
# rather than of this application's configuration.
ENV_VAR = "MEMCORE_ENV"

# An unwrapped master key is exactly one AES-256 key; a password-wrapped one is JSON.
RAW_KEY_SIZE = 32

class MasterPasswordRequired(RuntimeError):
    """Raised when a password-protected key must be unlocked but no password is available."""

class MasterKeyUnreadable(RuntimeError):
    """The key file exists but is neither a raw key nor a password-wrapped one."""

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

    @staticmethod
    def _parse_wrapped(raw: bytes):
        """Return the Argon2id wrapper dict, or None if this isn't a wrapped key."""
        try:
            data = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if isinstance(data, dict) and {'salt', 'nonce', 'ct'} <= data.keys():
            return data
        return None

    def load_or_create(self, password: str = None) -> bytes:
        """Load the master key, creating one only when no key file exists yet.

        The branch used to be ``key_path.exists() and stat().st_size > 100``. A raw,
        unprotected key is 32 bytes, so every unprotected store fell through to the
        *create* branch, which generated a fresh key and wrote it over the old one -
        silently making everything already encrypted undecryptable, while the process
        that happened to still hold the old key in memory kept working as if nothing
        were wrong. Decide on content, and never overwrite a key file that exists.
        """
        if self.key_path.exists():
            self._key = self._load(password)
            return self._key
        return self._create(password)

    def _load(self, password: str = None) -> bytes:
        raw = self.key_path.read_bytes()
        wrapped = self._parse_wrapped(raw)
        if wrapped is not None:
            salt = bytes.fromhex(wrapped['salt'])
            password = self._resolve_password(password)
            kdf = Argon2id(salt=salt, length=32, iterations=3, lanes=4, memory_cost=64*1024)
            kek = kdf.derive(password.encode())
            aes = AES256GCM(kek)
            return aes.decrypt(bytes.fromhex(wrapped['nonce']), bytes.fromhex(wrapped['ct']))
        if len(raw) == RAW_KEY_SIZE:
            return raw
        raise MasterKeyUnreadable(
            f"{self.key_path} is {len(raw)} bytes - neither a {RAW_KEY_SIZE}-byte raw key nor a "
            f"password-wrapped JSON key. Refusing to replace it, because that would make any "
            f"store encrypted under it unreadable. Move it aside to start a new store."
        )

    def _create(self, password: str = None) -> bytes:
        # A key written without a password sits on disk in the clear: anyone who can
        # read the file can read the store. That is a reasonable default for a local
        # dev box and not one for a deployment, so production refuses it outright.
        # Only creation is gated - an existing unprotected key still loads, otherwise
        # setting this would lock a running deployment out of its own data.
        if not password and os.environ.get(ENV_VAR, "").lower() in ("prod", "production"):
            raise ValueError(
                f"master password required: {ENV_VAR}={os.environ[ENV_VAR]} refuses to create an "
                f"unprotected master key at {self.key_path}. Pass --password or set "
                f"{MASTER_PASSWORD_ENV}."
            )
        master_key = AES256GCM.generate_key()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        if password:
            salt = os.urandom(16)
            kdf = Argon2id(salt=salt, length=32, iterations=3, lanes=4, memory_cost=64*1024)
            kek = kdf.derive(password.encode())
            aes = AES256GCM(kek)
            nonce, ct = aes.encrypt(master_key)
            self.key_path.write_text(json.dumps({'salt': salt.hex(), 'nonce': nonce.hex(), 'ct': ct.hex()}))
        else:
            self.key_path.write_bytes(master_key)
        os.chmod(self.key_path, 0o600)
        self._key = master_key
        return master_key

    @property
    def key(self) -> bytes:
        if not self._key:
            raise RuntimeError("Key not loaded")
        return self._key

    def get_cipher(self) -> AES256GCM:
        return AES256GCM(self.key)
