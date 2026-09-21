
"""
Token bucket rate limiting - excellent version
- Per-IP 100 req/min for /recall, 20 req/min for /add
- In-memory with sliding window, production would use Redis
"""
import time
from collections import defaultdict, deque
from fastapi import Request, HTTPException

class TokenBucket:
    def __init__(self, rate: int, per: int = 60):
        self.rate = rate
        self.per = per
        self.buckets = defaultdict(deque)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        q = self.buckets[key]
        # Remove old
        while q and q[0] <= now - self.per:
            q.popleft()
        if len(q) >= self.rate:
            return False
        q.append(now)
        return True

    def get_retry_after(self, key: str) -> int:
        q = self.buckets[key]
        if not q:
            return 0
        oldest = q[0]
        return int((oldest + self.per) - time.time()) + 1

# Global limiters
recall_limiter = TokenBucket(rate=100, per=60)  # 100 recall/min per IP
write_limiter = TokenBucket(rate=20, per=60)   # 20 writes/min per IP

async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    ip = request.client.host if request.client else "unknown"
    
    if "/recall" in path or "/search" in path:
        if not recall_limiter.is_allowed(f"{ip}:recall"):
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded for recall. Retry after {recall_limiter.get_retry_after(f'{ip}:recall')}s")
    elif "/add" in path or "/memory" in path and request.method == "POST":
        if not write_limiter.is_allowed(f"{ip}:write"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded for writes")
    
    return await call_next(request)
