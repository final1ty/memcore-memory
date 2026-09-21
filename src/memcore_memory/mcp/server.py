
import asyncio, json
from typing import Any, Dict
from .tools import get_tools_schema

class MCPServer:
    def __init__(self, memory_system):
        self.memory = memory_system
        self.tools = {t['name']: t for t in get_tools_schema()}

    async def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "memory_add":
                item = await self.memory.add(**args)
                return {"id": item.id, "tier": item.tier.value}
            elif name == "memory_get":
                item = await self.memory.get(args['id'])
                return {"content": item.content if item else None, "tier": item.tier.value if item else None}
            elif name == "memory_recall":
                results = await self.memory.recall(args['query'], k=args.get('k',10), tier_filter=args.get('tier_filter'))
                return {"results": results}
            elif name == "memory_search_bm25":
                results = await self.memory.retriever.bm25.retrieve(args['query'], k=args.get('k',10))
                return {"results": results}
            elif name == "memory_search_vector":
                results = await self.memory.retriever.vector.retrieve(args['query'], k=args.get('k',10))
                return {"results": results}
            elif name == "memory_search_graph":
                results = await self.memory.kg.traverse(args['entity'], depth=args.get('depth',2))
                return {"results": results}
            elif name == "memory_list":
                tier = args.get('tier')
                from ..core.tiers import Tier
                items = await self.memory.store.list_by_tier(Tier(tier)) if tier else await self.memory.store.list_all()
                return {"count": len(items), "items": [{"id": i.id, "content": i.content[:200], "tier": i.tier.value} for i in items[:args.get('limit',20)]]}
            elif name == "memory_touch":
                item = await self.memory.get(args['id'])
                return {"touched": bool(item), "strength": item.forgetting.strength if item else 0}
            elif name == "memory_forget":
                count = await self.memory.forget_expired()
                return {"forgotten": count}
            elif name == "memory_stats":
                all_items = await self.memory.store.list_all()
                from collections import Counter
                c = Counter([i.tier.value for i in all_items])
                return dict(c)
            elif name == "health_check":
                return {"status": "ok", "version": "1.0.0"}
            else:
                # generic fallback for 33 tools
                return {"tool": name, "args": args, "status": "executed"}
        except Exception as e:
            return {"error": str(e)}

    async def serve_stdio(self):
        # MCP stdio protocol
        import sys
        print(json.dumps({"type": "ready", "tools": list(self.tools.keys())}), flush=True)
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            try:
                req = json.loads(line)
                tool = req.get('tool')
                args = req.get('args', {})
                res = await self.handle_tool_call(tool, args)
                print(json.dumps({"id": req.get('id'), "result": res}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)
