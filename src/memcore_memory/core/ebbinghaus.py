
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
        # Clamped on read rather than trusted: importance -1 made S zero, so the
        # memory was deleted by the next forget pass, and anything above 1 made it
        # effectively immortal. Stored values are left as they are.
        imp = self.importance
        imp = 0.5 if imp != imp else min(1.0, max(0.0, imp))
        return self.strength * (1 + math.log1p(self.rehearsals)) * (1 + imp)

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

    def rehearse(self, now: float = None, feedback: float = 1.0, count: bool = True):
        """Strengthen the trace. `feedback` in [0,1] scales how much a recall counts.

        A recall the user confirmed was useful should stick harder than one that
        merely happened; feedback=1.0 reproduces the original S = S*1.6 + 0.5.
        count=False strengthens without adding to `rehearsals`, which is what tier
        promotion counts: a lower-ranked search hit is weaker evidence of use than
        a direct read, and counting it in full let a few loose searches promote.
        """
        now = now or time.time()
        feedback = max(0.0, min(1.0, feedback))
        self.strength = self.strength * (1 + 0.6 * feedback) + 0.5 * feedback
        if count:
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
        """Rebuild from storage, tolerating rows written before a field existed.

        Every field is coerced here and a value that cannot be is an error naming
        the field. Copying values through unchecked let a row with a string
        strength load cleanly and then fail inside retention() on every list and
        recall, while health and export kept reporting ok. Missing fields still
        take their defaults; a wrong one is never silently replaced, because the
        next put would persist the invented value over the original.
        """
        data = data or {}

        def num(name, default, typ=float):
            v = data.get(name)
            if v is None:
                return default
            try:
                return typ(v)
            except (TypeError, ValueError):
                raise ValueError(f"forgetting_json.{name} is not a {typ.__name__}: {v!r}") from None

        model = data.get("decay_model") or EXPONENTIAL
        if model not in (EXPONENTIAL, POWER_LAW):
            raise ValueError(f"forgetting_json.decay_model is unknown: {model!r}")
        return cls(
            strength=num("strength", 1.0),
            last_access=num("last_access", default_last_access),
            rehearsals=num("rehearsals", 0, int),
            importance=num("importance", 0.5),
            decay_model=model,
            power_d=num("power_d", 0.3),
        )
