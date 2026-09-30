"""Validated retry settings and a process-wide single-flight request gate."""
import math
import threading
import time
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime

@dataclass(frozen=True)
class RequestPolicy:
    # Retries exclude the initial request: five retries mean at most six calls.
    max_retries: int = 5
    request_interval_seconds: float = 1.0
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 32.0
    max_concurrency: int = 1

    def __post_init__(self):
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if type(self.max_concurrency) is not int or self.max_concurrency != 1:
            raise ValueError("LIBERO rollouts require max_concurrency=1")
        for name in ("request_interval_seconds", "initial_backoff_seconds", "max_backoff_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.initial_backoff_seconds <= 0 or self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("Require 0 < initial_backoff_seconds <= max_backoff_seconds")


def retry_after_seconds(value):
    """Accept seconds or an HTTP date; invalid headers fall back to backoff."""
    if value is None:
        return None
    try:
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, date.timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return None


class RequestGate:
    """Serialize all agents in this process, including full streamed responses."""
    def __init__(self):
        self.lock = threading.Lock()
        self.last_start = None
        self.previous_interval = 0.0

    def wait_for_start(self, interval):
        # Caller holds lock through the request AND any retry backoff.
        if self.last_start is not None:
            deadline = self.last_start + max(interval, self.previous_interval)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(remaining)
        self.last_start = time.monotonic()
        self.previous_interval = interval


REQUEST_GATE = RequestGate()
