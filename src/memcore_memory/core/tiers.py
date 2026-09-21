
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

@dataclass
class MemoryItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    tier: Tier = Tier.EPISODIC
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[list[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    forgetting: ForgettingCurve = field(default_factory=ForgettingCurve)
    entities: list[str] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.forgetting.last_access == 0:
            self.forgetting.last_access = self.timestamp
        if self.tier == Tier.SENSORY:
            self.forgetting.strength = 0.0005
        elif self.tier == Tier.WORKING:
            self.forgetting.strength = 0.014
        elif self.tier == Tier.EPISODIC:
            self.forgetting.strength = 7.0
        elif self.tier == Tier.SEMANTIC:
            self.forgetting.strength = 365.0

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
