
import asyncio, uuid, json
from pathlib import Path
from typing import List
from .crdt import MemoryCRDT
from .gossip import GossipProtocol
import websockets

class P2PNode:
    def __init__(self, node_id: str = None, port: int = 7742, peers: List[str] = None, store=None):
        self.node_id = node_id or str(uuid.uuid4())
        self.port = port
        self.peers = peers or []
        self.store = store
        self.crdt = MemoryCRDT(self.node_id)
        self.gossip = GossipProtocol(self.crdt, self.peers)
        self.server = None

    async def start(self):
        # start websocket server for sync
        self.server = await websockets.serve(self._handler, "0.0.0.0", self.port)
        print(f"[P2P] Node {self.node_id} listening on {self.port}")
        asyncio.create_task(self.gossip.gossip_loop())

    async def _handler(self, ws):
        async for msg in ws:
            try:
                data = json.loads(msg)
                if data.get('type') == 'sync':
                    # merge
                    # simplified: just ack
                    await ws.send(json.dumps({'type': 'ack', 'node': self.node_id}))
                elif data.get('type') == 'memory':
                    # receive memory
                    if self.store:
                        # store it
                        pass
            except Exception as e:
                await ws.send(json.dumps({'error': str(e)}))

    async def broadcast_memory(self, mem_id: str, content: dict):
        self.crdt.update(mem_id, content)
        # broadcast to peers via ws
        for peer in self.peers:
            try:
                async with websockets.connect(peer) as ws:
                    await ws.send(json.dumps({'type': 'memory', 'id': mem_id, 'data': content, 'node': self.node_id}))
            except Exception:
                continue

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.gossip.stop()
