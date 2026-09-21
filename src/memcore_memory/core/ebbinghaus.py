
import math, time
from dataclasses import dataclass

@dataclass
class ForgettingCurve:
    strength: float = 1.0
    last_access: float = 0.0
    rehearsals: int = 0
    importance: float = 0.5

    def retention(self, now: float = None) -> float:
        now = now or time.time()
        t = (now - self.last_access) / 86400.0
        if t <= 0:
            return 1.0
        S = self.strength * (1 + math.log1p(self.rehearsals)) * (1 + self.importance)
        R = math.exp(-t / S)
        return max(0.0, min(1.0, R))

    def rehearse(self, now: float = None):
        now = now or time.time()
        self.strength = self.strength * 1.6 + 0.5
        self.rehearsals += 1
        self.last_access = now

    def should_forget(self, threshold: float = 0.1, now: float = None) -> bool:
        return self.retention(now) < threshold

    def decay_weight(self, now: float = None) -> float:
        return self.retention(now)
