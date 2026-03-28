"""
Rate Limiter
============
Problem:
    Implement a rate limiter that enforces N requests per time window W.
    Support: is_allowed(user_id) → bool.

Two production-grade algorithms implemented:

Algorithm A — Token Bucket:
    - Tokens refill at rate R tokens/second up to capacity C.
    - Each request consumes 1 token. Allowed iff tokens >= 1.
    - Handles burst: a user can consume up to C requests instantly if bucket is full.
    - Used by: AWS API Gateway, Stripe.

Algorithm B — Sliding Window Counter:
    - Track request timestamps in a deque per user.
    - Evict timestamps older than window. Count remaining.
    - Allowed iff count < limit.
    - Exact (no burst tolerance). Memory: O(limit) per user.
    - Used by: Redis-based rate limiting (ZADD + ZREMRANGEBYSCORE pattern).

Algorithm C — Fixed Window Counter (simplest, shown for comparison):
    - Count requests per fixed epoch slot (floor(time / window)).
    - Boundary problem: allows 2× limit straddling a window edge.
    - Shown as a reference; prefer sliding window in production.

Edge cases:
    - Clock goes backward (monotonic clock assumed; assert if violated).
    - Multiple users (per-user buckets/windows).
    - Concurrent requests (thread-safety note: use locks in prod).

Complexity:
    Token Bucket   — Time: O(1) is_allowed, Space: O(U) for U users
    Sliding Window — Time: O(R) is_allowed where R = requests in window, Space: O(U * limit)
    Fixed Window   — Time: O(1), Space: O(U)
"""

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Dict


# ---------------------------------------------------------------------------
# Algorithm A: Token Bucket
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """Per-user token bucket rate limiter.

    Each user starts with a full bucket. Tokens refill continuously
    (lazy evaluation on each call).

    Args:
        capacity:    Max tokens (burst size).
        refill_rate: Tokens added per second.

    Raises:
        ValueError: If capacity or refill_rate <= 0.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if refill_rate <= 0:
            raise ValueError(f"refill_rate must be > 0, got {refill_rate}")
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._buckets: Dict[str, float] = defaultdict(lambda: capacity)
        self._last_refill: Dict[str, float] = defaultdict(time.monotonic)
        self._lock = Lock()

    def is_allowed(self, user_id: str) -> bool:
        """Check if user_id may make a request.

        Args:
            user_id: Identifier for the requester.

        Returns:
            True if request is allowed; False if rate-limited.

        Complexity:
            Time:  O(1)
            Space: O(1) amortized per call
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill[user_id]
            # Lazy refill: add tokens proportional to elapsed time
            self._buckets[user_id] = min(
                self._capacity,
                self._buckets[user_id] + elapsed * self._refill_rate,
            )
            self._last_refill[user_id] = now

            if self._buckets[user_id] >= 1.0:
                self._buckets[user_id] -= 1.0
                return True
            return False


# ---------------------------------------------------------------------------
# Algorithm B: Sliding Window Counter (exact)
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Per-user sliding window rate limiter (exact, timestamp-based).

    Maintains a deque of request timestamps per user.
    Evicts entries older than the window on every call.

    Args:
        limit:      Max requests allowed within the window.
        window_sec: Window duration in seconds.

    Raises:
        ValueError: If limit < 1 or window_sec <= 0.
    """

    def __init__(self, limit: int, window_sec: float) -> None:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if window_sec <= 0:
            raise ValueError(f"window_sec must be > 0, got {window_sec}")
        self._limit = limit
        self._window_sec = window_sec
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, user_id: str) -> bool:
        """Check if user_id may make a request.

        Args:
            user_id: Identifier for the requester.

        Returns:
            True if request is allowed; False if rate-limited.

        Complexity:
            Time:  O(R) where R = requests in current window (amortized O(1))
            Space: O(limit) per user
        """
        with self._lock:
            now = time.monotonic()
            window = self._windows[user_id]
            cutoff = now - self._window_sec

            # Evict timestamps outside the window
            while window and window[0] <= cutoff:
                window.popleft()

            if len(window) < self._limit:
                window.append(now)
                return True
            return False


# ---------------------------------------------------------------------------
# Algorithm C: Fixed Window Counter (reference — boundary problem)
# ---------------------------------------------------------------------------

class FixedWindowRateLimiter:
    """Per-user fixed window rate limiter.

    CAUTION: Allows up to 2× limit at window boundaries.
    Prefer SlidingWindowRateLimiter for production use.

    Args:
        limit:      Max requests per window.
        window_sec: Window duration in seconds.
    """

    def __init__(self, limit: int, window_sec: float) -> None:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if window_sec <= 0:
            raise ValueError(f"window_sec must be > 0, got {window_sec}")
        self._limit = limit
        self._window_sec = window_sec
        # (user_id → (window_slot, count))
        self._state: Dict[str, tuple] = {}
        self._lock = Lock()

    def is_allowed(self, user_id: str) -> bool:
        """Check if user_id may make a request.

        Complexity:
            Time:  O(1)
            Space: O(U) for U users
        """
        with self._lock:
            now = time.monotonic()
            slot = int(now // self._window_sec)
            prev_slot, count = self._state.get(user_id, (slot, 0))

            if slot != prev_slot:
                # New window: reset counter
                count = 0

            if count < self._limit:
                self._state[user_id] = (slot, count + 1)
                return True
            return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_token_bucket() -> None:
    # 2 req/sec, burst of 3
    rl = TokenBucketRateLimiter(capacity=3, refill_rate=2.0)
    # Burst: first 3 allowed
    assert rl.is_allowed("u1") is True
    assert rl.is_allowed("u1") is True
    assert rl.is_allowed("u1") is True
    # Bucket empty: 4th denied
    assert rl.is_allowed("u1") is False
    # Different user: independent bucket
    assert rl.is_allowed("u2") is True
    print("  TokenBucketRateLimiter: all tests passed")


def _test_sliding_window() -> None:
    # 3 requests per 1 second
    rl = SlidingWindowRateLimiter(limit=3, window_sec=1.0)
    assert rl.is_allowed("u1") is True
    assert rl.is_allowed("u1") is True
    assert rl.is_allowed("u1") is True
    assert rl.is_allowed("u1") is False   # 4th denied
    assert rl.is_allowed("u2") is True    # independent user
    print("  SlidingWindowRateLimiter: all tests passed")


def _test_fixed_window() -> None:
    rl = FixedWindowRateLimiter(limit=2, window_sec=1.0)
    assert rl.is_allowed("u1") is True
    assert rl.is_allowed("u1") is True
    assert rl.is_allowed("u1") is False
    print("  FixedWindowRateLimiter: all tests passed")


if __name__ == "__main__":
    print("Rate Limiter tests")
    _test_token_bucket()
    _test_sliding_window()
    _test_fixed_window()
