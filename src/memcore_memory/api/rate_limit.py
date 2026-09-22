
"""
Token bucket rate limiting - excellent version
- Per-IP 100 req/min for /recall, 20 req/min for /add
- In-memory with sliding window, production would use Redis
"""
import time
from collections import defaultdict, deque
from fastapi import Request
from fastapi.responses import JSONResponse

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

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


async def rate_limit_middleware(request: Request, call_next):
    """Limit reads and writes per client IP.

    Two things were wrong here. `"/add" in path or "/memory" in path and method ==
    "POST"` binds as `"/add" in path or ("/memory" in path and POST)`, because `and`
    binds tighter than `or` - so any GET to a path containing "/add" counted as a
    write, while DELETE /memory/{id} counted as nothing at all. And the middleware
    was never registered on the app, so none of it ran in the first place.

    Raising HTTPException from middleware does not produce a 429 either: Starlette
    only translates it inside the routing layer. Return the response directly.
    """
    path = request.url.path
    ip = request.client.host if request.client else "unknown"

    if "/recall" in path or "/search" in path:
        key = f"{ip}:recall"
        if not recall_limiter.is_allowed(key):
            retry = recall_limiter.get_retry_after(key)
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded for recall. Retry after {retry}s"},
                headers={"Retry-After": str(retry)},
            )
    elif request.method in WRITE_METHODS and ("/add" in path or "/memory" in path):
        key = f"{ip}:write"
        if not write_limiter.is_allowed(key):
            retry = write_limiter.get_retry_after(key)
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded for writes. Retry after {retry}s"},
                headers={"Retry-After": str(retry)},
            )

    return await call_next(request)
