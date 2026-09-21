
import httpx
from typing import List, Optional

class MnemosyneClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=30.0, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    def add(self, content: str, tier: str = None, importance: float = 0.5, entities: List[str] = None, metadata: dict = None) -> dict:
        resp = self.client.post(f"{self.base_url}/memory", json={"content": content, "tier": tier, "importance": importance, "entities": entities or [], "metadata": metadata or {}})
        resp.raise_for_status()
        return resp.json()

    def get(self, mem_id: str) -> dict:
        resp = self.client.get(f"{self.base_url}/memory/{mem_id}")
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
        resp = self.client.delete(f"{self.base_url}/memory/{mem_id}")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        resp = self.client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def consolidate(self):
        return self.client.post(f"{self.base_url}/consolidate").json()

    def forget_expired(self):
        return self.client.post(f"{self.base_url}/forget").json()
