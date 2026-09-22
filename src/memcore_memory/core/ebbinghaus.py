
import math, time
from dataclasses import dataclass
from typing import Any, Dict

EXPONENTIAL = "exponential"
POWER_LAW = "power_law"

@dataclass
class ForgettingCurve:
    strength: float = 1.0
    last_access: float = 0.0
    rehearsals: int = 0
    importance: float = 0.5
    # Ebbinghaus' own data fits an exponential; later replications (Wixted &
    # Ebbesen) fit a power law better over long intervals, because it keeps a
    # long tail instead of collapsing to zero. Both are offered; exponential
    # stays the default so nothing already stored changes behaviour.
    decay_model: str = EXPONENTIAL
    power_d: float = 0.3

    def effective_strength(self) -> float:
        """Stability S, after rehearsal count and importance are folded in."""
        return self.strength * (1 + math.log1p(self.rehearsals)) * (1 + self.importance)

    def retention(self, now: float = None) -> float:
        now = now or time.time()
        t = (now - self.last_access) / 86400.0
        if t <= 0:
            return 1.0
        S = self.effective_strength()
        if S <= 0:
            return 0.0
        if self.decay_model == POWER_LAW:
            R = (1.0 + t / S) ** (-self.power_d)
        else:
            R = math.exp(-t / S)
        return max(0.0, min(1.0, R))

    def rehearse(self, now: float = None, feedback: float = 1.0):
        """Strengthen the trace. `feedback` in [0,1] scales how much a recall counts.

        A recall the user confirmed was useful should stick harder than one that
        merely happened; feedback=1.0 reproduces the original S = S*1.6 + 0.5.
        """
        now = now or time.time()
        feedback = max(0.0, min(1.0, feedback))
        self.strength = self.strength * (1 + 0.6 * feedback) + 0.5 * feedback
        self.rehearsals += 1
        self.last_access = now

    def should_forget(self, threshold: float = 0.1, now: float = None) -> bool:
        return self.retention(now) < threshold

    def decay_weight(self, now: float = None) -> float:
        return self.retention(now)

    def to_dict(self) -> Dict[str, Any]:
        """What the storage layers persist.

        Every field that changes the curve belongs here. `decay_model` and
        `power_d` in particular: leave them out and a power-law memory silently
        comes back exponential on the next read, which is the same class of bug
        as the rehearsals that used to be discarded on load.
        """
        return {
            "strength": self.strength,
            "last_access": self.last_access,
            "rehearsals": self.rehearsals,
            "importance": self.importance,
            "decay_model": self.decay_model,
            "power_d": self.power_d,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_last_access: float = 0.0) -> "ForgettingCurve":
        """Rebuild from storage, tolerating rows written before a field existed."""
        data = data or {}
        return cls(
            strength=data.get("strength", 1.0),
            last_access=data.get("last_access", default_last_access),
            rehearsals=data.get("rehearsals", 0),
            importance=data.get("importance", 0.5),
            decay_model=data.get("decay_model", EXPONENTIAL),
            power_d=data.get("power_d", 0.3),
        )
