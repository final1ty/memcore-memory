
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class MemoryAddRequest(BaseModel):
    content: str
    tier: Optional[str] = None
    importance: float = 0.5
    entities: List[str] = []
    metadata: Dict[str, Any] = {}

class MemoryAddResponse(BaseModel):
    id: str
    tier: str

class RecallRequest(BaseModel):
    query: str
    k: int = 10
    tier_filter: Optional[List[str]] = None

class RecallResponse(BaseModel):
    results: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    version: str
    tier_counts: Dict[str, int]

class MCPCallRequest(BaseModel):
    """One MCP tool invocation forwarded by mcp/remote.py."""
    name: str
    arguments: Dict[str, Any] = {}
