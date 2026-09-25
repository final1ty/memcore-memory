
import os
import httpx
from typing import List, Optional
from urllib.parse import quote


def _seg(value: str) -> str:
    # One path segment: an id or entity holding '/', '?' or '#' ('C#', 'a/b')
    # otherwise became a different route or a query string.
    return quote(str(value), safe='')


def _connection(base_url: Optional[str], api_key: Optional[str]):
    """Resolve the endpoint and auth headers shared by both clients.

    There is deliberately no default URL. It used to be http://localhost:8000,
    which on the SkyNAS host is the live production container, so an example or
    a test run with no arguments wrote into the real store.
    """
    base_url = base_url or os.environ.get("MNEM_REMOTE_URL")
    if not base_url:
        raise ValueError("base_url is required (or set MNEM_REMOTE_URL), e.g. "
                         "MnemosyneClient('http://127.0.0.1:8000')")
    if api_key is None:
        api_key = os.environ.get("MNEM_API_KEY") or None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return base_url.rstrip('/'), headers


class MnemosyneClient:
    """Synchronous client for the memcore REST API.

    base_url: the REST instance, e.g. http://127.0.0.1:8000. Falls back to
        $MNEM_REMOTE_URL; there is no built-in default.
    api_key: sent as ``Authorization: Bearer <key>``, falling back to
        $MNEM_API_KEY. It only has an effect when the server sets MNEM_API_KEY;
        without that the REST API is unauthenticated and must not be reachable
        beyond localhost or a trusted LAN.
    """
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url, headers = _connection(base_url, api_key)
        self.client = httpx.Client(timeout=30.0, headers=headers)

    def add(self, content: str, tier: str = None, importance: float = 0.5, entities: List[str] = None, metadata: dict = None) -> dict:
        resp = self.client.post(f"{self.base_url}/memory", json={"content": content, "tier": tier, "importance": importance, "entities": entities or [], "metadata": metadata or {}})
        resp.raise_for_status()
        return resp.json()

    def get(self, mem_id: str) -> dict:
        resp = self.client.get(f"{self.base_url}/memory/{_seg(mem_id)}")
        resp.raise_for_status()
        return resp.json()

    def recall(self, query: str, k: int = 10, tier_filter: List[str] = None) -> List[dict]:
        resp = self.client.post(f"{self.base_url}/recall", json={"query": query, "k": k, "tier_filter": tier_filter})
        resp.raise_for_status()
        return resp.json()["results"]

    def list(self, tier: str = None, limit: int = 50) -> List[dict]:
        params = {"tier": tier, "limit": limit} if tier else {"limit": limit}
        resp = self.client.get(f"{self.base_url}/memories", params=params)
        resp.raise_for_status()
        return resp.json()

    def delete(self, mem_id: str) -> dict:
        resp = self.client.delete(f"{self.base_url}/memory/{_seg(mem_id)}")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        resp = self.client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    # These two returned the body of any error response - a 404 from a wrong
    # base_url, a proxy's 429 - as if it were the result.
    def consolidate(self) -> dict:
        resp = self.client.post(f"{self.base_url}/consolidate")
        resp.raise_for_status()
        return resp.json()

    def forget_expired(self) -> dict:
        resp = self.client.post(f"{self.base_url}/forget")
        resp.raise_for_status()
        return resp.json()

    def kg_traverse(self, entity: str, depth: int = 2, limit: int = 20) -> List[dict]:
        resp = self.client.get(f"{self.base_url}/kg/traverse/{_seg(entity)}", params={"depth": depth, "limit": limit})
        resp.raise_for_status()
        return resp.json()["results"]

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
