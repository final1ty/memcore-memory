
from typing import List

# P2P sync is not implemented: nothing transports or applies a peer's state, and
# /sync/merge answers 501. This used to poll it every 5 s, drop every non-200
# without a word and discard the body of a 200, so the loop looked healthy while
# doing nothing.
NOT_IMPLEMENTED = ("gossip sync is not implemented: /sync/merge has no auth model and no "
                   "response contract (it returns 501), so there is nothing to merge")


class GossipProtocol:
    def __init__(self, crdt, peers: List[str], interval: float = 5.0):
        # Shared by reference with P2PNode - merge into it with crdt.merge_from(),
        # never by rebinding self.crdt.
        self.crdt = crdt
        self.peers = peers
        self.interval = interval
        self.running = False

    async def gossip_loop(self):
        # Refuse up front rather than hit a 501 on a timer forever.
        raise NotImplementedError(NOT_IMPLEMENTED)

    async def _sync_with_peer(self, peer_url: str):
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{peer_url}/sync/merge", json=self.crdt.to_dict(), timeout=10.0)
            # A 501/404/500 used to fall through silently.
            resp.raise_for_status()
            # A 2xx body has no defined shape yet; guessing at one would merge
            # whatever an unauthenticated peer sent.
            raise NotImplementedError(NOT_IMPLEMENTED)

    def stop(self):
        self.running = False
