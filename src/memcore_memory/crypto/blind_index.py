"""
Blind Index for encrypted search - excellent version
- HMAC-SHA256 of normalized keywords using separate blind index key
- Allows search without decrypting content
- Per OWASP: use separate key from encryption key
"""
import hmac, hashlib, re, unicodedata
from typing import List, Set

# Bumped whenever _tokenize changes what it emits. Rows indexed under an older
# version don't error, they just stop matching, so the store records which
# version its index was built with (see EncryptedStore.search_by_blind_index).
TOKENIZER_VERSION = 2

STOPWORDS = {'the','and','for','are','but','not','you','all','can','her','was','one','our','out','day','get','has','him','his','how','its','may','new','now','old','see','two','way','who','boy','did','she','use','your','this','that','with','have','from','they','will','what','when','where','would','there','their'}


def _normalize(text: str) -> str:
    # NFC so a precomposed and a decomposed "á" hash alike; casefold rather than
    # lower so the query side and the index side agree on every script.
    return unicodedata.normalize('NFC', text).casefold()


class BlindIndex:
    def __init__(self, key: bytes):
        # The store passes a key derived with AES256GCM.derive_subkey, never the
        # content key itself.
        self.key = key
        if len(key) < 32:
            # Derive 32-byte key if needed
            self.key = hashlib.sha256(key).digest()

    def _tokenize(self, text: str) -> Set[str]:
        # This pattern MUST stay a raw string. It was once written without the r
        # prefix, so Python read each \b as a backspace (0x08) and the regex matched
        # nothing, ever: every blind index written was an empty list.
        #
        # [^\W_] is "a unicode letter or digit". The previous ASCII-only [a-z0-9]
        # split words at every accented letter and dropped the short fragments, so
        # "kész" or "őrző" produced no token at all and "tűzfal" became "zfal".
        # Underscore is excluded on purpose so "user_name" still yields both halves.
        tokens = re.findall(r'[^\W_]{3,}', _normalize(text))
        return {t for t in tokens if t not in STOPWORDS}

    @staticmethod
    def _tokenize_v1(text: str) -> Set[str]:
        """The pre-unicode tokenizer, kept only to query rows indexed with it."""
        tokens = re.findall(r'[a-z0-9]{3,}', text.lower())
        return {t for t in tokens if t not in STOPWORDS}

    def compute_token_hmac(self, token: str) -> str:
        # HMAC-SHA256(token) hex
        return hmac.new(self.key, token.encode(), hashlib.sha256).hexdigest()

    def compute_index(self, content: str, metadata: dict = None) -> List[str]:
        # Compute blind index for content + metadata
        tokens = self._tokenize(content)
        if metadata:
            meta_text = " ".join(str(v) for v in metadata.values())
            tokens |= self._tokenize(meta_text)
        # Return list of HMACs
        return sorted([self.compute_token_hmac(t) for t in tokens])

    def search_token_hmac(self, query_token: str) -> str:
        return self.compute_token_hmac(_normalize(query_token))

    def search_query_hmacs(self, query: str, include_legacy: bool = False) -> List[str]:
        """HMACs to look up for `query`.

        With `include_legacy`, the old tokenizer's tokens are added too, so rows
        indexed before the unicode change keep matching until they are reindexed.
        """
        tokens = self._tokenize(query)
        if include_legacy:
            tokens |= self._tokenize_v1(query)
        return [self.compute_token_hmac(t) for t in tokens]
