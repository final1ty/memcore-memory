
import httpx
from typing import List, Optional
from .client import _connection, _seg

class AsyncMnemosyneClient:
    """The async twin of MnemosyneClient: same methods, same requests, awaited.

    It used to offer only add and recall. base_url and api_key behave as in
    MnemosyneClient: no default URL ($MNEM_REMOTE_URL is the fallback), and the
    key is sent as ``Authorization: Bearer <key>`` ($MNEM_API_KEY), which only
    matters when the server sets MNEM_API_KEY. Without it the REST API is
    unauthenticated and must not be reachable beyond localhost or a trusted LAN.
    """
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url, headers = _connection(base_url, api_key)
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)

    async def add(self, content: str, tier: str = None, importance: float = 0.5, entities: List[str] = None, metadata: dict = None) -> dict:
        resp = await self.client.post(f"{self.base_url}/memory", json={"content": content, "tier": tier, "importance": importance, "entities": entities or [], "metadata": metadata or {}})
        resp.raise_for_status()
        return resp.json()

    async def get(self, mem_id: str) -> dict:
        resp = await self.client.get(f"{self.base_url}/memory/{_seg(mem_id)}")
        resp.raise_for_status()
        return resp.json()

    async def recall(self, query: str, k: int = 10, tier_filter: List[str] = None) -> List[dict]:
        resp = await self.client.post(f"{self.base_url}/recall", json={"query": query, "k": k, "tier_filter": tier_filter})
        resp.raise_for_status()
        return resp.json()["results"]

    async def list(self, tier: str = None, limit: int = 50) -> List[dict]:
        params = {"tier": tier, "limit": limit} if tier else {"limit": limit}
        resp = await self.client.get(f"{self.base_url}/memories", params=params)
        resp.raise_for_status()
        return resp.json()

    async def delete(self, mem_id: str) -> dict:
        resp = await self.client.delete(f"{self.base_url}/memory/{_seg(mem_id)}")
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> dict:
        resp = await self.client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    async def consolidate(self) -> dict:
        resp = await self.client.post(f"{self.base_url}/consolidate")
        resp.raise_for_status()
        return resp.json()

    async def forget_expired(self) -> dict:
        resp = await self.client.post(f"{self.base_url}/forget")
        resp.raise_for_status()
        return resp.json()

    async def kg_traverse(self, entity: str, depth: int = 2, limit: int = 20) -> List[dict]:
        resp = await self.client.get(f"{self.base_url}/kg/traverse/{_seg(entity)}", params={"depth": depth, "limit": limit})
        resp.raise_for_status()
        return resp.json()["results"]

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
