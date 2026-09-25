"""Claude Desktop MCP bridge to the SkyNAS memcore container (MCP name "memcore-skynas").

Lives on the Windows machine as C:\\Users\\A\\memcore_remote_mcp.py; this copy is the
source. The container requires an API key once MNEM_API_KEY is set there: the key is read
from %MEMCORE_API_KEY% or, failing that, from C:\\Users\\A\\.memcore-api-key (first line).
"""
import os
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("memcore-skynas")
BASE = os.environ.get("MEMCORE_URL", "http://192.168.1.183:8000")
KEY_FILE = Path.home() / ".memcore-api-key"


def _key():
    key = os.environ.get("MEMCORE_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    return key or None


def _client():
    key = _key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.Client(base_url=BASE, headers=headers, timeout=10)


def _json(r):
    if r.status_code == 401:
        raise RuntimeError(f"memcore rejected the API key (401): check {KEY_FILE} or MEMCORE_API_KEY")
    r.raise_for_status()
    return r.json()


@mcp.tool()
def health():
    # "degraded" is HTTP 200 and still serves; only 503 means unhealthy.
    with _client() as c:
        r = c.get("/health")
        return r.json() if r.status_code in (200, 503) else _json(r)


@mcp.tool()
def recall(query: str, k: int = 5):
    with _client() as c:
        return _json(c.post("/recall", json={"query": query, "k": k}))


@mcp.tool()
def add_memory(content: str, tier: str = "working"):
    with _client() as c:
        return _json(c.post("/memory", json={"content": content, "tier": tier}))


@mcp.tool()
def list_memories(limit: int = 50):
    with _client() as c:
        return _json(c.get("/memories", params={"limit": limit}))


if __name__ == "__main__":
    mcp.run()
