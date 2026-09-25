
import asyncio, contextlib, uuid, json
from typing import List
from .crdt import MemoryCRDT
from .gossip import GossipProtocol

# Nothing here transports or applies memories. The node used to bind 0.0.0.0
# with no auth, answer "ack" to a sync without merging, drop received memories
# and swallow unreachable peers - a demo that "succeeded" while moving nothing.
# Every entry point now says so instead. Wiring a real merge into an
# unauthenticated socket would recreate the open LAN write endpoint /sync/merge
# was turned into a 501 to avoid, so that needs an auth model first.
NOT_IMPLEMENTED = ("P2P sync is not implemented: there is no authenticated transport, "
                   "and received state is never merged or stored")


class P2PNode:
    def __init__(self, node_id: str = None, port: int = 7742, peers: List[str] = None, store=None,
                 host: str = "127.0.0.1"):
        self.node_id = node_id or str(uuid.uuid4())
        self.host = host
        self.port = port
        self.peers = peers or []
        self.store = store
        self.crdt = MemoryCRDT(self.node_id)
        self.gossip = GossipProtocol(self.crdt, self.peers)
        self.server = None
        self._gossip_task = None

    async def start(self):
        raise NotImplementedError(NOT_IMPLEMENTED)

    async def _handler(self, ws):
        # Unreachable while start() refuses, but it must never acknowledge a sync
        # it did not perform.
        async for msg in ws:
            try:
                kind = json.loads(msg).get('type')
            except Exception:
                kind = None
            if kind in ('sync', 'memory'):
                reply = {'type': 'error', 'error': NOT_IMPLEMENTED}
            else:
                reply = {'type': 'error', 'error': f'unknown message type {kind!r}'}
            await ws.send(json.dumps(reply))

    async def broadcast_memory(self, mem_id: str, content: dict):
        # No peer would store it, and failures used to be swallowed.
        raise NotImplementedError(NOT_IMPLEMENTED)

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.gossip.stop()
        if self._gossip_task is not None:
            self._gossip_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._gossip_task
            self._gossip_task = None
