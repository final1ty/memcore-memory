
"""
Blind Index for encrypted search - excellent version
- HMAC-SHA256 of normalized keywords using separate blind index key
- Allows search without decrypting content
- Per OWASP: use separate key from encryption key
"""
import hmac, hashlib, re
from typing import List, Set
from .aes_gcm import AES256GCM

class BlindIndex:
    def __init__(self, key: bytes):
        # Blind index key should be separate from encryption key - derive via HKDF
        self.key = key
        if len(key) < 32:
            # Derive 32-byte key if needed
            self.key = hashlib.sha256(key).digest()

    @staticmethod
    def derive_blind_key(master_key: bytes) -> bytes:
        # HKDF: blind_index_key = HKDF(master_key, info=b"blind-index")
        return hmac.new(master_key, b"blind-index-v1", hashlib.sha256).digest()

    def _tokenize(self, text: str) -> Set[str]:
        # Normalize: lower, alphanumeric, min 3 chars, stopwords removed
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]{3,}', text)
        stopwords = {'the','and','for','are','but','not','you','all','can','her','was','one','our','out','day','get','has','him','his','how','its','may','new','now','old','see','two','way','who','boy','did','she','use','your','this','that','with','have','from','they','will','what','when','where','would','there','their'}
        return {t for t in tokens if t not in stopwords}

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
        return self.compute_token_hmac(query_token.lower())

    def search_query_hmacs(self, query: str) -> List[str]:
        tokens = self._tokenize(query)
        return [self.compute_token_hmac(t) for t in tokens]
