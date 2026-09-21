
import asyncio
from mnemosyne.sync.p2p import P2PNode

async def main():
    node1 = P2PNode(port=7742, peers=[])
    node2 = P2PNode(port=7743, peers=["ws://localhost:7742"])
    
    await node1.start()
    await node2.start()
    
    await node1.broadcast_memory("mem-123", {"content": "Federated memory"})
    
    await asyncio.sleep(2)
    await node1.stop()
    await node2.stop()

asyncio.run(main())
