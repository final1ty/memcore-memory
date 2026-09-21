
import httpx
from typing import List

class AsyncMnemosyneClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    async def add(self, content: str, tier: str = None, importance: float = 0.5, entities: List[str] = None, metadata: dict = None):
        resp = await self.client.post(f"{self.base_url}/memory", json={"content": content, "tier": tier, "importance": importance, "entities": entities or [], "metadata": metadata or {}})
        resp.raise_for_status()
        return resp.json()

    async def recall(self, query: str, k: int = 10, tier_filter: List[str] = None):
        resp = await self.client.post(f"{self.base_url}/recall", json={"query": query, "k": k, "tier_filter": tier_filter})
        resp.raise_for_status()
        return resp.json()["results"]

    async def close(self):
        await self.client.aclose()
