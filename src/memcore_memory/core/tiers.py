
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
            from ..config import settings
            self.forgetting = ForgettingCurve(
                strength=TIER_BASE_STRENGTH.get(self.tier, 7.0),
                importance=self.metadata.get('importance', 0.5),
                # Only new memories take the configured model; a loaded curve keeps
                # the one it was written with.
                decay_model=settings.forgetting_model,
            )
        else:
            # Importance is kept twice: in metadata, which users set and see, and on
            # the curve, which drives decay. memory_update changes only the first, so
            # a memory reported as important went on decaying as unimportant. The
            # metadata copy wins; nothing else on a loaded curve is touched.
            imp = _importance_or_none(self.metadata.get('importance'))
            if imp is not None:
                self.forgetting.importance = imp
        if self.forgetting.last_access == 0:
            self.forgetting.last_access = self.timestamp

    def touch(self):
        self.forgetting.rehearse()

    def set_importance(self, value) -> float:
        """Change importance in both places it is kept, rejecting out-of-range values."""
        imp = validate_importance(value)
        self.metadata['importance'] = imp
        self.forgetting.importance = imp
        return imp


def validate_importance(value) -> float:
    """Importance is a weight in [0, 1]; anything else is a caller error, not a clamp.

    Below -1 the stability went to zero or negative and the memory was deleted by
    the next forget pass; above 1 it became effectively immortal. The MCP schema
    advertised the range but nothing enforced it.
    """
    # bool is an int subclass, so True used to pass as 1.0 - the most important a
    # memory can be - from a caller that meant a flag. _importance_or_none agrees.
    if isinstance(value, bool):
        raise ValueError(f"importance must be a number in [0, 1], got {value!r}")
    try:
        imp = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"importance must be a number in [0, 1], got {value!r}") from None
    if not 0.0 <= imp <= 1.0:  # also false for NaN
        raise ValueError(f"importance must be in [0, 1], got {value!r}")
    return imp


def _importance_or_none(value):
    if isinstance(value, bool):
        return None
    try:
        imp = float(value)
    except (TypeError, ValueError):
        return None
    if imp != imp:
        return None
    return min(1.0, max(0.0, imp))

class TierManager:
    """Every rule that moves a memory between tiers or deletes it lives here.

    The lifecycle, decided once:

    - SENSORY: deleted once older than ``sensory_ttl_seconds``, unless it was marked
      ``attended`` and is still retained (R > 0.3), in which case it moves up to
      WORKING.
    - WORKING: a staging area, never a place memories are deleted from. New memories
      without an explicit tier land here. A working memory moves down to EPISODIC
      when it has been rehearsed, when it is older than ``working_ttl_seconds``, or
      when newer ones push the tier past ``working_capacity`` (MnemosyneMemory
      enforces the cap against the store, oldest first, so it holds across
      processes and restarts). Deleting on expiry used to lose an explicit
      tier='working' memory within the hour. Expiry is applied by the lifecycle
      pass and, opportunistically, by every add; until one of them runs, a working
      memory past its TTL keeps its short working-tier curve and reports a low
      retention. Demotion raises its strength to the episodic floor.
    - EPISODIC: deleted when retention falls below 0.05. Promoted to SEMANTIC only
      when all three hold: at least ``semantic_consolidation_threshold`` rehearsals,
      still well retained (R > 0.6), and importance above
      ``semantic_min_importance``. Rehearsal alone is not enough, because recall
      rehearses its top results and three searches would otherwise make anything
      permanent.
    - SEMANTIC: never deleted automatically. Only an explicit delete removes it.

    Promotion is judged before deletion and wins: a memory that has earned a move
    up is never deleted in the same pass.
    """

    def __init__(self, settings):
        self.settings = settings

    def assign_tier(self, item: MemoryItem, context: dict = None) -> Tier:
        if context and context.get('tier'):
            return Tier(context['tier'])
        if item.metadata.get('sensory'):
            return Tier.SENSORY
        if (time.time() - item.timestamp) < self.settings.working_ttl_seconds:
            return Tier.WORKING
        return Tier.EPISODIC

    def should_promote(self, item: MemoryItem):
        R = item.forgetting.retention()
        if item.tier == Tier.SENSORY and R > 0.3 and item.metadata.get('attended'):
            return Tier.WORKING
        if item.tier == Tier.WORKING and item.forgetting.rehearsals >= 1:
            return Tier.EPISODIC
        if (item.tier == Tier.EPISODIC
                and item.forgetting.rehearsals >= self.settings.semantic_consolidation_threshold
                and R > 0.6
                and item.forgetting.importance > self.settings.semantic_min_importance):
            return Tier.SEMANTIC
        return None

    def should_demote_or_forget(self, item: MemoryItem):
        now = time.time()
        if item.tier == Tier.SENSORY:
            if (now - item.timestamp) > self.settings.sensory_ttl_seconds:
                return "forget"
            return None
        if item.tier == Tier.WORKING:
            if (now - item.timestamp) > self.settings.working_ttl_seconds:
                return "demote"
            return None
        if item.tier == Tier.EPISODIC and item.forgetting.should_forget(threshold=0.05, now=now):
            return "forget"
        return None
