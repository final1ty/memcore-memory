# MCP server demo - 29 tools, called in-process. Runs against a throwaway store in a
# fresh temp directory, set before the package is imported because config.py reads
# the environment at import time.
import asyncio
import os
import tempfile

_demo_dir = tempfile.mkdtemp(prefix="memcore-mcp-demo-")
# Overwrite, not setdefault: MNEM_DATA_DIR may already point at a live volume.
os.environ["MNEM_DATA_DIR"] = _demo_dir
for _var in ("MNEM_DB_PATH", "MNEM_KEY_PATH", "MNEM_VECTOR_PATH", "MNEM_AUDIT_LOG_PATH",
             "MNEM_WORKING_BUFFER_PATH", "MNEM_BACKEND", "MNEM_DATABASE_URL",
             "MNEM_MASTER_PASSWORD", "MNEM_ENV", "MEMCORE_ENV"):
    os.environ.pop(_var, None)

from memcore_memory import create_memory_system  # noqa: E402  (after the env setup)
from memcore_memory.mcp.server import MCPServer  # noqa: E402


async def main():
    mem = await create_memory_system()
    server = MCPServer(mem)

    # Episodic rather than semantic: even if this ever ran against a real store, the
    # row would decay instead of lasting for years.
    print(await server.handle_tool_call("memory_add", {"content": "MCP tool test", "tier": "episodic", "importance": 0.5}))
    print(await server.handle_tool_call("memory_recall", {"query": "MCP test", "k": 5}))
    print(await server.handle_tool_call("memory_stats", {}))
    print(f"Demo store: {_demo_dir}")


if __name__ == "__main__":
    asyncio.run(main())
