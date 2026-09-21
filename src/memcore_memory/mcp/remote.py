"""MCP server that proxies to a running memcore REST API instead of opening a store.

Why this exists. Under the Docker deployment the encrypted store lives inside the
container and its named volume is root-owned, so a stdio MCP server started on the
host cannot open that database at all. The host's own ``~/.memcore`` *is* openable -
which is the trap: it is a completely separate store that diverged from the one the
REST API serves, so an MCP client pointed at it answers confidently from stale data.
There is no path where two processes reading two different SQLite files are one
memory system.

This bridge removes the second store from the picture. It advertises the same tool
surface as the local server and forwards every call to ``POST /mcp/call`` on the
running instance, which dispatches through the exact same handlers. One store, one
source of truth, and no master password on the client side because the server
unlocked its key at startup.

One semantic difference to know about: ``memory_export`` and ``memory_import`` take a
filesystem path, and that path is resolved by the *server*. Over this bridge they read
and write inside the container, not on the machine running the MCP client.
"""

import sys
from typing import Any, Dict, Optional

from .server import INSTRUCTIONS, SERVER_NAME, StdioMCPServer, ToolError


class RemoteUnavailable(RuntimeError):
    """The configured endpoint is unreachable or too old to expose the bridge."""


class RemoteMCPServer(StdioMCPServer):
    """Advertises the remote instance's tools and forwards calls over HTTP."""

    def __init__(self, base_url: str, client=None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None
        # Tools come from the server in connect(); start empty so a stale client
        # can never advertise a tool the remote does not actually implement.
        super().__init__([], INSTRUCTIONS)

    async def connect(self):
        """Fetch the remote tool list. Raises RemoteUnavailable with a usable diagnosis."""
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        try:
            response = await self._client.get("/mcp/tools")
        except httpx.HTTPError as e:
            raise RemoteUnavailable(f"cannot reach {self.base_url}: {e}") from e
        if response.status_code == 404:
            raise RemoteUnavailable(
                f"{self.base_url} answers, but has no /mcp/tools endpoint. That instance "
                f"predates the REST-backed MCP bridge - rebuild and recreate it "
                f"(docker compose up -d --build)."
            )
        response.raise_for_status()
        payload = response.json()
        super().__init__(payload["tools"], INSTRUCTIONS)
        print(
            f"[{SERVER_NAME}] proxying {len(self.tools)} tools to {self.base_url} "
            f"(remote version {payload.get('version', 'unknown')})",
            file=sys.stderr, flush=True,
        )
        return self

    async def aclose(self):
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def handle_tool_call(self, name: str, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        import httpx

        if self._client is None:
            raise ToolError("bridge is not connected; call connect() first")
        try:
            response = await self._client.post(
                "/mcp/call", json={"name": name, "arguments": args or {}}
            )
        except httpx.HTTPError as e:
            raise ToolError(f"{self.base_url} unreachable: {e}") from e
        if response.status_code >= 500:
            raise ToolError(f"{self.base_url} returned {response.status_code}: {response.text[:200]}")
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise ToolError(payload.get("error", "remote call failed without a reason"))
        return payload["result"]

    async def serve_stdio(self):
        try:
            await super().serve_stdio()
        finally:
            await self.aclose()
