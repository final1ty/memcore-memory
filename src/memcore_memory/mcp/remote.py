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
file name that the *server* resolves inside its own ``<data_dir>/exports``. Over this
bridge they read and write inside the container, not on the machine running the MCP
client.

When the server sets ``MNEM_API_KEY``, the bridge must send the same key: pass
``api_key`` or set ``MNEM_API_KEY`` on the client side as well.
"""

import sys
from typing import Any, Dict, Optional

from .server import INSTRUCTIONS, SERVER_NAME, StdioMCPServer, ToolError


def _describe(e: Exception) -> str:
    # str() of several httpx errors is empty; the type name alone is still a reason.
    text = str(e)
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


class RemoteUnavailable(RuntimeError):
    """The configured endpoint is unreachable or too old to expose the bridge."""


class RemoteMCPServer(StdioMCPServer):
    """Advertises the remote instance's tools and forwards calls over HTTP."""

    def __init__(self, base_url: str, client=None, timeout: float = 30.0, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None
        if api_key is None:
            from ..config import settings
            api_key = getattr(settings, "api_key", None)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Tools come from the server in connect(); start empty so a stale client
        # can never advertise a tool the remote does not actually implement.
        super().__init__([], INSTRUCTIONS)

    async def connect(self):
        """Fetch the remote tool list. Raises RemoteUnavailable with a usable diagnosis."""
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        try:
            payload = await self._fetch_tools()
        except BaseException:
            # A failed connect used to leave the client it had just opened behind.
            await self.aclose()
            raise
        super().__init__(payload["tools"], INSTRUCTIONS)
        print(
            f"[{SERVER_NAME}] proxying {len(self.tools)} tools to {self.base_url} "
            f"(remote version {payload.get('version', 'unknown')})",
            file=sys.stderr, flush=True,
        )
        return self

    async def _fetch_tools(self) -> Dict[str, Any]:
        # Every way the answer can be wrong gets its own diagnosis. A bare 404 used
        # to be reported as "predates the bridge - rebuild the container" even for a
        # wrong port (another service entirely), and a 200 that was not memcore's
        # escaped as a raw KeyError or JSONDecodeError traceback.
        import httpx

        try:
            response = await self._client.get("/mcp/tools", headers=self._headers)
        except httpx.HTTPError as e:
            raise RemoteUnavailable(f"cannot reach {self.base_url}: {_describe(e)}") from e
        if response.status_code == 404:
            if await self._looks_like_memcore():
                raise RemoteUnavailable(
                    f"{self.base_url} answers, but has no /mcp/tools endpoint. That instance "
                    f"predates the REST-backed MCP bridge - rebuild and recreate it "
                    f"(docker compose up -d --build)."
                )
            raise RemoteUnavailable(
                f"{self.base_url} answered 404 for /mcp/tools and does not look like a memcore "
                f"REST API - check the host, port and path (--remote / MNEM_REMOTE_URL)."
            )
        if response.status_code == 401:
            raise RemoteUnavailable(
                f"{self.base_url} requires an API key and "
                f"{'rejected the one sent' if self._headers else 'none was sent'} - "
                f"set MNEM_API_KEY to the server's key.")
        if response.status_code != 200:
            raise RemoteUnavailable(
                f"{self.base_url} answered HTTP {response.status_code} for /mcp/tools: {response.text[:200]!r}")
        try:
            payload = response.json()
        except ValueError:
            raise RemoteUnavailable(f"{self.base_url} answered /mcp/tools with something other than JSON; "
                                    f"it is not a memcore instance") from None
        if not isinstance(payload, dict) or payload.get("server") != SERVER_NAME \
                or not isinstance(payload.get("tools"), list):
            server = payload.get("server") if isinstance(payload, dict) else None
            raise RemoteUnavailable(f"{self.base_url} is not a memcore instance (server={server!r})")
        return payload

    async def _looks_like_memcore(self) -> bool:
        try:
            r = await self._client.get("/health", headers=self._headers)
            j = r.json()
        except Exception:  # noqa: BLE001 - any failure means "no"
            return False
        # A degraded or failing instance is still memcore; the 503 is its own report.
        return (r.status_code in (200, 503) and isinstance(j, dict)
                and j.get("status") in ("ok", "degraded", "unhealthy") and "version" in j)

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
                "/mcp/call", json={"name": name, "arguments": args or {}}, headers=self._headers
            )
        # Order matters: ConnectTimeout and PoolTimeout are TimeoutExceptions, and
        # all of these are HTTPErrors. Only a failed connect, or no free connection
        # in the pool, means the call never left this process. A read timeout used
        # to be reported as "unreachable: " with an empty reason while the server
        # finished the write, so a retry did it twice.
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            raise ToolError(f"{self.base_url} unreachable, the call was not delivered: {_describe(e)}") from e
        except httpx.TimeoutException as e:
            raise ToolError(
                f"{self.base_url} did not answer {name!r} within {self.timeout}s ({type(e).__name__}). "
                f"The request was sent and may still complete on the server - check with "
                f"health_check or memory_list before retrying, or raise --timeout.") from e
        except httpx.HTTPError as e:
            raise ToolError(f"{self.base_url} request failed: {_describe(e)}") from e
        if response.status_code == 401:
            raise ToolError(f"{self.base_url} rejected the call: missing or invalid API key (MNEM_API_KEY)")
        if response.status_code == 429:
            raise ToolError(f"{self.base_url} rate-limited {name!r}; retry after "
                            f"{response.headers.get('Retry-After', '?')}s")
        if response.status_code >= 400:
            # A 422 from request validation, or a 404 from a server without the
            # route, used to escape as a raw HTTPStatusError.
            raise ToolError(f"{self.base_url} returned {response.status_code} for {name!r}: "
                            f"{response.text[:200]}")
        try:
            payload = response.json()
        except ValueError:
            raise ToolError(f"{self.base_url} answered {name!r} with something other than JSON: "
                            f"{response.text[:200]!r}") from None
        if not isinstance(payload, dict):
            raise ToolError(f"{self.base_url} answered {name!r} with {type(payload).__name__}, not an object")
        if not payload.get("ok"):
            raise ToolError(payload.get("error", "remote call failed without a reason"))
        return payload["result"]

    async def serve_stdio(self):
        try:
            await super().serve_stdio()
        finally:
            await self.aclose()
