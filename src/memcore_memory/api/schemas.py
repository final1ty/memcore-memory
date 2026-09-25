from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

from ..core.tiers import Tier

# Bounded so a bad value is a 422 naming the field. An unknown tier used to be a
# 500 from Tier(...) in the handler, importance=99 was stored and made the memory
# effectively immortal, and k=-1 sliced from the end and returned unrelated rows.

class MemoryAddRequest(BaseModel):
    content: str
    tier: Optional[Tier] = None
    importance: float = Field(0.5, ge=0.0, le=1.0)
    entities: List[str] = []
    metadata: Dict[str, Any] = {}

class MemoryAddResponse(BaseModel):
    id: str
    tier: str

class RecallRequest(BaseModel):
    query: str
    k: int = Field(10, ge=1, le=100)  # the same bound as the MCP tools' k
    tier_filter: Optional[List[Tier]] = None
    # The top 3 matched results are rehearsed (their retention is strengthened)
    # unless this is false - the same rule as the MCP memory_recall tool.
    rehearse: bool = True

class RecallResponse(BaseModel):
    results: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str  # "ok", "degraded" (200, see reasons) or "unhealthy" (503)
    version: str
    tier_counts: Dict[str, int]
    reasons: List[str] = []
    unreadable_count: int = 0
    unreadable: List[str] = []  # the first HEALTH_MAX_IDS ids only; unreadable_count is exact

class MCPCallRequest(BaseModel):
    """One MCP tool invocation forwarded by mcp/remote.py."""
    name: str
    arguments: Dict[str, Any] = {}
