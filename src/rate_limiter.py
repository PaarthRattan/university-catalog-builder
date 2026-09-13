"""Client-side enforcement of Gemini's three rate limits: RPM, TPM and RPD.

The API enforces all three independently and returns 429 RESOURCE_EXHAUSTED when
any one is exceeded. This module enforces the same three locally so we spend our
quota on useful work instead of on rejected requests.

RPM and TPM are rolling windows held in memory. RPD is a rolling 24h window
persisted to disk, because a daily budget necessarily outlives a single process
run -- without persistence, restarting the pipeline would silently reset our
view of the day's spend and walk straight into a server-side 429.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DAY_SECONDS = 24 * 60 * 60
MINUTE_SECONDS = 60


class RateLimitExceeded(Exception):
    """Local equivalent of a 429: a limit would be exceeded by this request."""

    def __init__(self, limit_name: str, retry_after: float, message: str = None):
        self.limit_name = limit_name
        self.retry_after = retry_after
        super().__init__(message or
                         f"{limit_name} limit would be exceeded; retry in {retry_after:.1f}s")


class DailyQuotaExhausted(Exception):
    """The requests-per-day budget is spent. Not retryable within this window."""

    def __init__(self, used: int, limit: int, resets_in: float):
        self.used = used
        self.limit = limit
        self.resets_in = resets_in
        hours = resets_in / 3600.0
        super().__init__(
            f"Daily Gemini quota exhausted: {used}/{limit} requests used. "
            f"Oldest request ages out in {hours:.1f}h."
        )


class GeminiRateLimiter:
    """Thread-safe RPM + TPM + RPD limiter.

    acquire() blocks until the per-minute limits permit the request, and raises
    DailyQuotaExhausted the moment the daily budget is gone (waiting hours for a
    daily window to roll is never the right behaviour -- the caller should
    checkpoint and exit instead).
    """

    def __init__(self, rpm: int, tpm: int, rpd: int,
                 state_path: Optional[str] = None,
                 name: str = "gemini"):
        if rpm < 1 or tpm < 1 or rpd < 1:
            raise ValueError("rpm, tpm and rpd must all be >= 1")
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.name = name
        self.state_path = state_path

        self._lock = threading.Condition(threading.Lock())
        # (timestamp, tokens) for the trailing minute
        self._minute: Deque[Tuple[float, int]] = deque()
        # bare timestamps for the trailing day
        self._day: Deque[float] = deque()

        # Observability
        self.total_wait_seconds = 0.0
        self.rpm_blocks = 0
        self.tpm_blocks = 0
        # Real limits as reported by server 429s, if we ever see one.
        self.server_reported: Dict[str, str] = {}

        self._load_state()

    # ------------------------------------------------------------------ state

    def _load_state(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as fh:
                data = json.load(fh)
            cutoff = time.time() - DAY_SECONDS
            stamps = [float(t) for t in data.get("day_requests", []) if float(t) > cutoff]
            self._day = deque(sorted(stamps))
            if self._day:
                logger.info("Loaded RPD state: %d requests in the trailing 24h (limit %d)",
                            len(self._day), self.rpd)
        except Exception as exc:  # a corrupt state file must not block the run
            logger.warning("Could not read rate-limit state %s: %s", self.state_path, exc)

    def _save_state(self):
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            tmp = f"{self.state_path}.tmp"
            with open(tmp, "w") as fh:
                json.dump({"day_requests": list(self._day)}, fh)
            os.replace(tmp, self.state_path)
        except Exception as exc:
            logger.warning("Could not persist rate-limit state: %s", exc)

    # ------------------------------------------------------------- internals

    def _prune(self, now: float):
        minute_cutoff = now - MINUTE_SECONDS
        while self._minute and self._minute[0][0] <= minute_cutoff:
            self._minute.popleft()
        day_cutoff = now - DAY_SECONDS
        while self._day and self._day[0] <= day_cutoff:
            self._day.popleft()

    def _tokens_in_window(self) -> int:
        return sum(tok for _, tok in self._minute)

    def _wait_needed(self, now: float, tokens: int) -> Tuple[float, Optional[str]]:
        """Seconds to wait before this request fits, and which limit binds."""
        if len(self._minute) >= self.rpm:
            return (self._minute[0][0] + MINUTE_SECONDS) - now, "RPM"
        if self._tokens_in_window() + tokens > self.tpm:
            # Retire just enough of the oldest entries to make room.
            need = self._tokens_in_window() + tokens - self.tpm
            freed = 0
            for ts, tok in self._minute:
                freed += tok
                if freed >= need:
                    return (ts + MINUTE_SECONDS) - now, "TPM"
            return MINUTE_SECONDS, "TPM"
        return 0.0, None

    # ------------------------------------------------------------------- API

    def check(self, estimated_tokens: int = 0) -> None:
        """Non-blocking check. Raises rather than waiting.

        Raises DailyQuotaExhausted if the daily budget is spent, or
        RateLimitExceeded if a per-minute limit currently binds.
        """
        with self._lock:
            now = time.time()
            self._prune(now)
            self._check_daily_locked(now)
            wait, which = self._wait_needed(now, estimated_tokens)
            if wait > 0:
                raise RateLimitExceeded(which, wait)

    def _check_daily_locked(self, now: float):
        if len(self._day) >= self.rpd:
            resets_in = (self._day[0] + DAY_SECONDS) - now
            raise DailyQuotaExhausted(len(self._day), self.rpd, max(0.0, resets_in))

    def acquire(self, estimated_tokens: int = 0) -> float:
        """Reserve one request slot, blocking on RPM/TPM. Returns seconds waited.

        Raises DailyQuotaExhausted immediately if the daily budget is gone.
        """
        waited = 0.0
        with self._lock:
            while True:
                now = time.time()
                self._prune(now)
                self._check_daily_locked(now)

                wait, which = self._wait_needed(now, estimated_tokens)
                if wait <= 0:
                    self._minute.append((now, max(0, estimated_tokens)))
                    self._day.append(now)
                    self._save_state()
                    self.total_wait_seconds += waited
                    return waited

                if which == "RPM":
                    self.rpm_blocks += 1
                else:
                    self.tpm_blocks += 1
                logger.debug("%s limiter: waiting %.2fs on %s", self.name, wait, which)
                # Condition.wait releases the lock, so other threads can drain.
                self._lock.wait(timeout=min(wait, 5.0))
                waited += min(wait, 5.0)

    def record_actual_tokens(self, estimated: int, actual: int):
        """Reconcile the pre-call estimate with the count the API reported."""
        delta = actual - estimated
        if delta == 0:
            return
        with self._lock:
            if self._minute:
                ts, tok = self._minute[-1]
                self._minute[-1] = (ts, max(0, tok + delta))
            self._lock.notify_all()

    def release_slot(self):
        """Give back a reservation for a request that was never actually sent."""
        with self._lock:
            if self._minute:
                self._minute.pop()
            if self._day:
                self._day.pop()
            self._save_state()
            self._lock.notify_all()

    def note_server_quota(self, quota_id: str, quota_value: str):
        """Record a limit the server told us about in a 429.

        The server is authoritative; if it disagrees with our configured limit we
        say so loudly, because it means config is wrong.
        """
        with self._lock:
            self.server_reported[quota_id] = quota_value
        try:
            value = int(quota_value)
        except (TypeError, ValueError):
            return
        qid = quota_id.lower()
        if "perminute" in qid and value != self.rpm:
            logger.warning("Server reports RPM limit of %d but config says %d. "
                           "Set GEMINI_RPM=%d in .env.", value, self.rpm, value)
        elif "perday" in qid and value != self.rpd:
            logger.warning("Server reports RPD limit of %d but config says %d. "
                           "Set GEMINI_RPD=%d in .env.", value, self.rpd, value)

    def snapshot(self) -> Dict:
        with self._lock:
            now = time.time()
            self._prune(now)
            return {
                "rpm_limit": self.rpm,
                "tpm_limit": self.tpm,
                "rpd_limit": self.rpd,
                "requests_last_minute": len(self._minute),
                "tokens_last_minute": self._tokens_in_window(),
                "requests_last_day": len(self._day),
                "daily_remaining": max(0, self.rpd - len(self._day)),
                "total_wait_seconds": round(self.total_wait_seconds, 3),
                "rpm_blocks": self.rpm_blocks,
                "tpm_blocks": self.tpm_blocks,
                "server_reported_quotas": dict(self.server_reported),
            }
