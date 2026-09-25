
from pathlib import Path
from .aes_gcm import AES256GCM
import errno, os, sys, json, getpass, sqlite3, tempfile, time
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

MASTER_PASSWORD_ENV = "MNEM_MASTER_PASSWORD"
# Deployment marker: it gates whether an unprotected key may be created at all,
# which is a property of the environment rather than of the store. MNEM_ENV
# matches every other variable this project reads; MEMCORE_ENV is the name it
# had before the MNEM_ prefix fix and is still honoured so nothing that already
# sets it silently loses the guard.
ENV_VAR = "MNEM_ENV"
LEGACY_ENV_VAR = "MEMCORE_ENV"

# An unwrapped master key is exactly one AES-256 key; a password-wrapped one is JSON.
RAW_KEY_SIZE = 32

# Bookkeeping tables. Every other table counts as data: an unknown table is
# assumed to hold rows that need the old key, which errs toward refusing.
_META_TABLES = ("store_meta", "kg_meta")

# A temp file younger than this may belong to a creator or protect() still running
# in another process; older ones are leftovers of a process that died mid-write.
_STALE_TMP_S = 60.0

# Filesystems without hard links (some CIFS/SMB mounts) report one of these from link().
_NO_HARDLINK_ERRNOS = {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV}


class MasterKeyError(RuntimeError):
    """Base for every reason the master key cannot be used. Catch this to report one line."""

class MasterPasswordRequired(MasterKeyError):
    """Raised when a password-protected key must be unlocked but no password is available."""

class WrongMasterPassword(MasterPasswordRequired):
    """The password was supplied but does not unwrap the key file."""

class UnprotectedKeyRefused(MasterPasswordRequired, ValueError):
    """Production refuses to create a key that would sit on disk in the clear.

    Also a ValueError, which is what this used to raise before it had a type.
    """

class MasterKeyUnreadable(MasterKeyError):
    """The key file exists but is neither a raw key nor a password-wrapped one."""

class MasterKeyMissing(MasterKeyError):
    """No key file, but a database next to it already holds data encrypted under one."""

class MasterKeyMismatch(MasterKeyError):
    """The key loads fine but is not the key the store was written under."""


def deployment_env() -> tuple:
    """(variable name, lower-cased value) of the deployment marker, or ('', '')."""
    for name in (ENV_VAR, LEGACY_ENV_VAR):
        value = os.environ.get(name)
        if value:
            return name, value.lower()
    return "", ""


def _fsync_dir(path: Path):
    # A new or replaced name is only durable once its directory entry is.
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class KeyManager:
    def __init__(self, key_path: Path):
        self.key_path = Path(key_path)
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

    @staticmethod
    def _kek(password: str, salt: bytes) -> AES256GCM:
        kdf = Argon2id(salt=salt, length=32, iterations=3, lanes=4, memory_cost=64*1024)
        return AES256GCM(kdf.derive(password.encode()))

    @classmethod
    def _wrap(cls, master_key: bytes, password: str) -> bytes:
        salt = os.urandom(16)
        nonce, ct = cls._kek(password, salt).encrypt(master_key)
        return json.dumps({'salt': salt.hex(), 'nonce': nonce.hex(), 'ct': ct.hex()}).encode()

    def _unwrap(self, wrapped: dict, password: str) -> bytes:
        aes = self._kek(password, bytes.fromhex(wrapped['salt']))
        try:
            return aes.decrypt(bytes.fromhex(wrapped['nonce']), bytes.fromhex(wrapped['ct']))
        except InvalidTag:
            # A bare InvalidTag names neither the file nor the password, so a typo looked
            # exactly like a corrupted key. Raise only: never fall through to creating a
            # key, which is how the key-destruction bug happened.
            raise WrongMasterPassword(
                f"the master password does not unlock {self.key_path} (wrong password, or "
                f"the wrapped key file is damaged). Check --password / {MASTER_PASSWORD_ENV}."
            ) from None

    def load_or_create(self, password: str = None, db_path: Path = None,
                       allow_new_key: bool = False) -> bytes:
        """Load the master key, creating one only when no key file exists yet.

        The branch used to be ``key_path.exists() and stat().st_size > 100``. A raw,
        unprotected key is 32 bytes, so every unprotected store fell through to the
        *create* branch, which generated a fresh key and wrote it over the old one -
        silently making everything already encrypted undecryptable, while the process
        that happened to still hold the old key in memory kept working as if nothing
        were wrong. Decide on content, and never overwrite a key file that exists.

        A missing key is not always a new store either: ``db_path`` (defaulting to the
        configured store when this is the configured key) is checked, and a key is not
        created next to a database that already holds data unless ``allow_new_key``.
        """
        # Resolved once, so creation honours MNEM_MASTER_PASSWORD the way loading always
        # did - library callers that set it used to get a key stored in the clear.
        password = password or os.environ.get(MASTER_PASSWORD_ENV) or None
        self._remove_stale_temp_files()
        if self.key_path.exists():
            self._key = self._load(password)
            return self._key
        if not allow_new_key:
            self._refuse_if_store_has_data(db_path)
        return self._create(password)

    def _temp_prefix(self) -> str:
        return f".{self.key_path.name}."

    def _remove_stale_temp_files(self):
        """Delete key temp files left by a process killed between write and cleanup.

        Each holds a full copy of the key (raw, or wrapped under a password that may
        since have changed), and backups and rescue copies of the data dir would
        carry it along. None is ever the only copy of a key in use: creation returns
        only after the key is linked into place, and protect() only replaces a key
        with the verified wrapped copy of itself.
        """
        parent = self.key_path.parent
        if not parent.is_dir():
            return
        now = time.time()
        prefix = self._temp_prefix()
        try:
            entries = list(parent.iterdir())
        except OSError:
            return
        for tmp in entries:
            if not (tmp.name.startswith(prefix) and tmp.name.endswith(".tmp")):
                continue
            try:
                if now - tmp.lstat().st_mtime > _STALE_TMP_S:
                    tmp.unlink()
                    print(f"[memcore] removed stale key temp file {tmp}", file=sys.stderr)
            except OSError:
                pass

    def _load(self, password: str = None) -> bytes:
        raw = self.key_path.read_bytes()
        wrapped = self._parse_wrapped(raw)
        if wrapped is not None:
            return self._unwrap(wrapped, self._resolve_password(password))
        if len(raw) == RAW_KEY_SIZE:
            if password:
                # Loading must never rewrite the key file, and must not start failing
                # because a password happens to be set - but a user who supplied one
                # believes the store is protected, so say plainly that it is not.
                print(f"[memcore] warning: {self.key_path} is an UNPROTECTED raw key; the "
                      f"supplied password was ignored and the file left unchanged. "
                      f"KeyManager.protect() wraps it in place.",
                      file=sys.stderr)
            return raw
        raise MasterKeyUnreadable(
            f"{self.key_path} is {len(raw)} bytes - neither a {RAW_KEY_SIZE}-byte raw key nor a "
            f"password-wrapped JSON key. Refusing to replace it, because that would make any "
            f"store encrypted under it unreadable. Move it aside to start a new store."
        )

    def _store_paths(self, db_path: Path = None) -> list:
        if db_path is None:
            try:
                from ..config import settings
                if settings.key_path and Path(settings.key_path).resolve() == self.key_path.resolve():
                    db_path = settings.db_path
            except Exception:
                db_path = None
        if db_path is None:
            return []
        db_path = Path(db_path)
        return [db_path, db_path.with_suffix('.kg.db')]

    @staticmethod
    def _rows_in(db_file: Path) -> int:
        """Rows across the data tables of an SQLite file, opened read-only."""
        if not db_file.exists() or db_file.stat().st_size == 0:
            return 0
        con = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
        try:
            # kg_meta has to be excluded too: KnowledgeGraph.init() writes its schema
            # version there, so an initialised but empty graph used to count as data
            # and block creating the first key.
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'") if r[0] not in _META_TABLES]
            return sum(con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables)
        finally:
            con.close()

    def _refuse_if_store_has_data(self, db_path: Path = None):
        # Runs before anything is written: a refused start must not leave a stray key
        # behind, or the next start would load it and write under the wrong key.
        for db_file in self._store_paths(db_path):
            try:
                rows = self._rows_in(db_file)
            except sqlite3.Error as e:
                raise MasterKeyMissing(
                    f"{self.key_path} does not exist and {db_file} cannot be inspected "
                    f"({type(e).__name__}: {e}); refusing to create a new key next to it. "
                    f"Restore the original master.key, or move {db_file} aside to start over."
                ) from None
            if rows:
                raise MasterKeyMissing(
                    f"{self.key_path} does not exist but {db_file} holds {rows} encrypted "
                    f"row(s). Restore the original master.key (or fix MNEM_KEY_PATH / "
                    f"MNEM_DATA_DIR) - a new key could never decrypt them, and anything "
                    f"written under it would be lost once the original is put back. Move the "
                    f"database aside to deliberately start a new store."
                )

    def _create(self, password: str = None) -> bytes:
        # A key written without a password sits on disk in the clear: anyone who can
        # read the file can read the store. That is a reasonable default for a local
        # dev box and not one for a deployment, so production refuses it outright.
        # Only creation is gated - an existing unprotected key still loads, otherwise
        # setting this would lock a running deployment out of its own data.
        env_name, env_value = deployment_env()
        if not password and env_value in ("prod", "production"):
            raise UnprotectedKeyRefused(
                f"master password required: {env_name}={env_value} refuses to create an "
                f"unprotected master key at {self.key_path}. Pass --password or set "
                f"{MASTER_PASSWORD_ENV}."
            )
        master_key = AES256GCM.generate_key()
        data = self._wrap(master_key, password) if password else master_key
        if not self._publish_exclusive(data):
            # Another process created the key between our exists() check and now. Its
            # key is the one on disk, so it is the only one that may be used: carrying
            # on with ours would encrypt rows under a key that exists nowhere.
            self._key = self._adopt(password)
            return self._key
        self._key = master_key
        return master_key

    def _publish_exclusive(self, data: bytes) -> bool:
        """Put `data` at key_path only if nothing is there. False if something was.

        Written to a private temp file (0600 from birth, so no world-readable window)
        and hard-linked into place: link() is atomic and refuses an existing name, so
        readers see a complete file or none, and two creators cannot both win.
        """
        parent = self.key_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=self._temp_prefix(), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, self.key_path)
            except FileExistsError:
                return False
            except OSError as e:
                if e.errno not in _NO_HARDLINK_ERRNOS:
                    raise
                if not self._write_exclusive(data):
                    return False
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        _fsync_dir(parent)
        return True

    def _write_exclusive(self, data: bytes) -> bool:
        # Fallback without hard links: O_EXCL still guarantees one winner, but a reader
        # can catch the file mid-write, which _adopt() tolerates by retrying.
        try:
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return True

    def _adopt(self, password: str = None, attempts: int = 40, delay: float = 0.05) -> bytes:
        for attempt in range(attempts):
            try:
                return self._load(password)
            except MasterKeyUnreadable:
                # Only a file this young can still be being written; an old unreadable
                # file is genuinely broken and must be reported, never replaced.
                young = time.time() - self.key_path.stat().st_mtime < 2.0
                if not young or attempt == attempts - 1:
                    raise
                time.sleep(delay)

    def protect(self, password: str) -> bytes:
        """Wrap an existing raw key with a password, keeping the key itself unchanged.

        The one legitimate rewrite of a key file, so it is only ever called by an
        explicit command, never as a side effect of loading. The wrapped file is
        read back and unwrapped before it replaces the raw one, and the raw file is
        re-checked right before the swap, so a failure at any point leaves the
        original in place.
        """
        if not password:
            raise ValueError("protect() needs a non-empty password")
        raw = self.key_path.read_bytes()
        if self._parse_wrapped(raw) is not None:
            raise MasterKeyError(f"{self.key_path} is already password-protected")
        if len(raw) != RAW_KEY_SIZE:
            raise MasterKeyUnreadable(
                f"{self.key_path} is {len(raw)} bytes - not a {RAW_KEY_SIZE}-byte raw key; "
                f"refusing to wrap it."
            )
        fd, tmp = tempfile.mkstemp(dir=self.key_path.parent, prefix=self._temp_prefix(),
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(self._wrap(raw, password))
                f.flush()
                os.fsync(f.fileno())
            written = self._parse_wrapped(Path(tmp).read_bytes())
            if written is None or self._unwrap(written, password) != raw:
                raise MasterKeyError(f"wrapped copy of {self.key_path} did not verify; left unchanged")
            if self.key_path.read_bytes() != raw:
                raise MasterKeyError(f"{self.key_path} changed while it was being wrapped; left unchanged")
            os.replace(tmp, self.key_path)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        _fsync_dir(self.key_path.parent)
        self._key = raw
        return raw

    def is_protected(self) -> bool:
        """True when the key file on disk is password-wrapped. Reads, never writes."""
        return self._parse_wrapped(self.key_path.read_bytes()) is not None

    @property
    def key(self) -> bytes:
        if not self._key:
            raise RuntimeError("Key not loaded")
        return self._key

    def get_cipher(self) -> AES256GCM:
        return AES256GCM(self.key)
