
import asyncio, json, random
from typing import List

class GossipProtocol:
    def __init__(self, crdt, peers: List[str], interval: float = 5.0):
        self.crdt = crdt
        self.peers = peers
        self.interval = interval
        self.running = False

    async def gossip_loop(self):
        self.running = True
        while self.running:
            await asyncio.sleep(self.interval)
            if not self.peers:
                continue
            peer = random.choice(self.peers)
            try:
                await self._sync_with_peer(peer)
            except Exception as e:
                print(f"[gossip] failed sync with {peer}: {e}")

    async def _sync_with_peer(self, peer_url: str):
        import httpx
        async with httpx.AsyncClient() as client:
            # push
            resp = await client.post(f"{peer_url}/sync/merge", json=self.crdt.to_dict(), timeout=10.0)
            if resp.status_code == 200:
                remote = resp.json()
                # merge remote into local (simplified)
                # In prod, deserialize remote CRDT and merge
                pass

    def stop(self):
        self.running = False
