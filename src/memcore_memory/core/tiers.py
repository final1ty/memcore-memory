
from enum import Enum
from dataclasses import dataclass, field
import time, uuid
from typing import Optional, Any, Dict
from .ebbinghaus import ForgettingCurve

class Tier(str, Enum):
    SENSORY = "sensory"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

# Starting Ebbinghaus strength S0 for a *new* memory in each tier, in days.
TIER_BASE_STRENGTH = {
    Tier.SENSORY: 0.0005,   # ~30s
    Tier.WORKING: 0.014,    # ~20min
    Tier.EPISODIC: 7.0,     # weeks
    Tier.SEMANTIC: 365.0,   # years
}

@dataclass
class MemoryItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    tier: Tier = Tier.EPISODIC
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[list[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # None means "new memory": seed the curve from the tier. A curve passed in
    # explicitly (i.e. loaded from storage) is authoritative and must not be
    # overwritten - doing so discarded every rehearsal the memory had earned and
    # left the forgetting curve permanently pinned to its tier baseline.
    forgetting: Optional[ForgettingCurve] = None
    entities: list[str] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.forgetting is None:
            self.forgetting = ForgettingCurve(
                strength=TIER_BASE_STRENGTH.get(self.tier, 7.0),
                importance=self.metadata.get('importance', 0.5),
            )
        if self.forgetting.last_access == 0:
            self.forgetting.last_access = self.timestamp

    def touch(self):
        self.forgetting.rehearse()

class TierManager:
    def __init__(self, settings):
        self.settings = settings
        self.working_buffer: list[MemoryItem] = []

    def assign_tier(self, item: MemoryItem, context: dict = None) -> Tier:
        if context and context.get('tier'):
            return Tier(context['tier'])
        imp = item.metadata.get('importance', 0.5)
        if item.metadata.get('sensory'):
            return Tier.SENSORY
        if imp > 0.8 and item.forgetting.rehearsals >= self.settings.semantic_consolidation_threshold:
            return Tier.SEMANTIC
        if item.forgetting.rehearsals >= 1 or (time.time() - item.timestamp) < self.settings.working_ttl_seconds:
            if len(self.working_buffer) < self.settings.working_capacity:
                return Tier.WORKING
        return Tier.EPISODIC

    def should_promote(self, item: MemoryItem):
        R = item.forgetting.retention()
        if item.tier == Tier.SENSORY and R > 0.3 and item.metadata.get('attended'):
            return Tier.WORKING
        if item.tier == Tier.WORKING and item.forgetting.rehearsals >= 1:
            return Tier.EPISODIC
        if item.tier == Tier.EPISODIC and item.forgetting.rehearsals >= self.settings.semantic_consolidation_threshold:
            return Tier.SEMANTIC
        return None

    def should_demote_or_forget(self, item: MemoryItem):
        now = time.time()
        if item.tier == Tier.SENSORY and (now - item.timestamp) > self.settings.sensory_ttl_seconds:
            return "forget"
        if item.tier == Tier.WORKING and (now - item.timestamp) > self.settings.working_ttl_seconds and item.forgetting.retention(now) < 0.2:
            return "forget"
        if item.forgetting.should_forget(threshold=0.05 if item.tier != Tier.SEMANTIC else 0.01):
            return "forget"
        return None
