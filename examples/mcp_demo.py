
# MCP Server demo - 33 tools
import asyncio
from mnemosyne import create_memory_system
from mnemosyne.mcp.server import MCPServer

async def main():
    mem = await create_memory_system()
    server = MCPServer(mem)
    
    # Simulate tool calls
    r1 = await server.handle_tool_call("memory_add", {"content": "MCP tool test", "tier": "semantic", "importance": 0.9})
    print(r1)
    r2 = await server.handle_tool_call("memory_recall", {"query": "MCP test", "k": 5})
    print(r2)
    r3 = await server.handle_tool_call("memory_stats", {})
    print(r3)

asyncio.run(main())
