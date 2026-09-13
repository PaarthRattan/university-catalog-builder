"""Structured timing and cost instrumentation for pipeline runs.

Every number this module reports is measured at the point of use -- there are no
estimates here except the dollar figure, which is an explicit multiplication of
measured token counts by configured per-token prices.
"""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class StageMetrics:
    """Counters for a single pipeline stage."""

    def __init__(self, name: str):
        self.name = name
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.items = 0              # pages (or universities) handled
        self.requests = 0           # successful Gemini requests
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.retries = 0            # retry attempts made
        self.rate_limit_hits = 0    # server 429s observed
        self.local_throttles = 0    # requests delayed by our own limiter
        self.failures = 0           # calls that exhausted retries
        self.wait_seconds = 0.0     # time blocked on rate limits
        self.api_seconds = 0.0      # time inside generate_content calls

    @property
    def wall_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else time.time()
        return end - self.started_at

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens

    def _per_minute(self, count: float) -> float:
        wall = self.wall_seconds
        return (count / wall * 60.0) if wall > 0 else 0.0

    def to_dict(self) -> Dict:
        wall = self.wall_seconds
        return {
            "stage": self.name,
            "wall_seconds": round(wall, 3),
            "items": self.items,
            "items_per_min": round(self._per_minute(self.items), 2),
            "gemini_requests": self.requests,
            "gemini_requests_per_min": round(self._per_minute(self.requests), 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
            "retries": self.retries,
            "rate_limit_429s": self.rate_limit_hits,
            "local_throttles": self.local_throttles,
            "failures": self.failures,
            "rate_limit_wait_seconds": round(self.wait_seconds, 3),
            "api_call_seconds": round(self.api_seconds, 3),
            "other_seconds": round(max(0.0, wall - self.wait_seconds - self.api_seconds), 3),
            # thread-seconds per 100s of wall clock; >100 means concurrent waiting
            "wait_threadsec_per_100s_wall": round(100.0 * self.wait_seconds / wall, 1) if wall > 0 else 0.0,
        }


class MetricsCollector:
    """Thread-safe collector; one per pipeline run."""

    def __init__(self, run_label: str, metrics_dir: str,
                 price_input_per_mtok: float = 0.0,
                 price_output_per_mtok: float = 0.0,
                 config_snapshot: Optional[Dict] = None):
        self.run_label = run_label
        self.metrics_dir = metrics_dir
        self.price_input = price_input_per_mtok
        self.price_output = price_output_per_mtok
        self.config_snapshot = config_snapshot or {}
        self.run_started_at = time.time()
        self.run_started_iso = datetime.now(timezone.utc).isoformat()
        self.run_ended_at: Optional[float] = None
        self.notes: List[str] = []
        self._stages: Dict[str, StageMetrics] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._current: Optional[str] = None

    # ---------------------------------------------------------------- stages

    def _stage(self, name: str) -> StageMetrics:
        if name not in self._stages:
            self._stages[name] = StageMetrics(name)
            self._order.append(name)
        return self._stages[name]

    @contextmanager
    def stage(self, name: str):
        with self._lock:
            st = self._stage(name)
            if st.started_at is None:
                st.started_at = time.time()
            prev, self._current = self._current, name
        try:
            yield st
        finally:
            with self._lock:
                st.ended_at = time.time()
                self._current = prev

    def current_stage(self) -> str:
        return self._current or "unattributed"

    # -------------------------------------------------------------- counters

    def record_items(self, count: int, stage: Optional[str] = None):
        with self._lock:
            self._stage(stage or self.current_stage()).items += count

    def record_request(self, input_tokens: int, output_tokens: int,
                       thinking_tokens: int = 0, api_seconds: float = 0.0,
                       wait_seconds: float = 0.0, retries: int = 0,
                       rate_limit_hits: int = 0, throttled: bool = False,
                       stage: Optional[str] = None):
        with self._lock:
            st = self._stage(stage or self.current_stage())
            st.requests += 1
            st.input_tokens += input_tokens
            st.output_tokens += output_tokens
            st.thinking_tokens += thinking_tokens
            st.api_seconds += api_seconds
            st.wait_seconds += wait_seconds
            st.retries += retries
            st.rate_limit_hits += rate_limit_hits
            if throttled:
                st.local_throttles += 1

    def record_failure(self, retries: int = 0, rate_limit_hits: int = 0,
                       wait_seconds: float = 0.0, stage: Optional[str] = None):
        with self._lock:
            st = self._stage(stage or self.current_stage())
            st.failures += 1
            st.retries += retries
            st.rate_limit_hits += rate_limit_hits
            st.wait_seconds += wait_seconds

    def add_note(self, note: str):
        with self._lock:
            self.notes.append(note)

    # --------------------------------------------------------------- totals

    def totals(self) -> Dict:
        stages = [self._stages[n] for n in self._order]
        inp = sum(s.input_tokens for s in stages)
        out = sum(s.output_tokens for s in stages)
        think = sum(s.thinking_tokens for s in stages)
        wall = (self.run_ended_at or time.time()) - self.run_started_at
        wait = sum(s.wait_seconds for s in stages)
        api = sum(s.api_seconds for s in stages)
        # Thinking tokens are billed at the output rate.
        cost = (inp / 1e6) * self.price_input + ((out + think) / 1e6) * self.price_output
        return {
            "run_wall_seconds": round(wall, 3),
            "gemini_requests": sum(s.requests for s in stages),
            "input_tokens": inp,
            "output_tokens": out,
            "thinking_tokens": think,
            "total_tokens": inp + out + think,
            "estimated_cost_usd": round(cost, 6),
            "pricing_configured": bool(self.price_input or self.price_output),
            "retries": sum(s.retries for s in stages),
            "rate_limit_429s": sum(s.rate_limit_hits for s in stages),
            "local_throttles": sum(s.local_throttles for s in stages),
            "failures": sum(s.failures for s in stages),
            "rate_limit_wait_seconds": round(wait, 3),
            "api_call_seconds": round(api, 3),
            "other_seconds": round(max(0.0, wall - wait - api), 3),
            "wait_threadsec_per_100s_wall": round(100.0 * wait / wall, 1) if wall > 0 else 0.0,
        }

    def to_dict(self, extra: Optional[Dict] = None) -> Dict:
        return {
            "run_label": self.run_label,
            "started_utc": self.run_started_iso,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config_snapshot,
            "stages": [self._stages[n].to_dict() for n in self._order],
            "totals": self.totals(),
            "notes": self.notes,
            **(extra or {}),
        }

    # --------------------------------------------------------------- output

    def finalize(self, extra: Optional[Dict] = None) -> str:
        self.run_ended_at = time.time()
        os.makedirs(self.metrics_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.run_label)
        path = os.path.join(self.metrics_dir, f"{safe}_{stamp}.json")
        payload = self.to_dict(extra)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Metrics written to %s", path)
        return path

    def format_summary(self) -> str:
        rows = [self._stages[n].to_dict() for n in self._order]
        t = self.totals()

        # Wait(s)/API(s) are thread-seconds summed across concurrent workers, so
        # they can exceed Wall(s) when concurrency > 1. Wall(s) is real elapsed time.
        headers = ["Stage", "Wall(s)", "Items", "Items/min", "Reqs", "Reqs/min",
                   "Tokens", "429s", "Retry", "Fail", "Wait(s)", "API(s)"]
        data = [[
            r["stage"], f"{r['wall_seconds']:.1f}", str(r["items"]),
            f"{r['items_per_min']:.1f}", str(r["gemini_requests"]),
            f"{r['gemini_requests_per_min']:.2f}", f"{r['total_tokens']:,}",
            str(r["rate_limit_429s"]), str(r["retries"]), str(r["failures"]),
            f"{r['rate_limit_wait_seconds']:.1f}", f"{r['api_call_seconds']:.1f}",
        ] for r in rows]
        data.append([
            "TOTAL", f"{t['run_wall_seconds']:.1f}", "", "",
            str(t["gemini_requests"]), "", f"{t['total_tokens']:,}",
            str(t["rate_limit_429s"]), str(t["retries"]), str(t["failures"]),
            f"{t['rate_limit_wait_seconds']:.1f}", f"{t['api_call_seconds']:.1f}",
        ])

        widths = [max(len(headers[i]), *(len(r[i]) for r in data)) for i in range(len(headers))]
        sep = "-+-".join("-" * w for w in widths)

        def line(cells):
            return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

        out = ["", "=" * len(sep), f"RUN METRICS: {self.run_label}", "=" * len(sep),
               line(headers), sep]
        out += [line(r) for r in data[:-1]]
        out += [sep, line(data[-1]), ""]

        out.append(f"Gemini requests : {t['gemini_requests']:,}")
        out.append(f"Input tokens    : {t['input_tokens']:,}")
        out.append(f"Output tokens   : {t['output_tokens']:,}"
                   + (f"  (+{t['thinking_tokens']:,} thinking)" if t["thinking_tokens"] else ""))
        if t["pricing_configured"]:
            out.append(f"Estimated cost  : ${t['estimated_cost_usd']:.4f}")
        else:
            out.append("Estimated cost  : $0.0000 (free tier; no prices configured)")
        out.append(f"429s / retries  : {t['rate_limit_429s']} / {t['retries']}")
        out.append(f"Failures        : {t['failures']}")
        out.append(f"Time breakdown  : {t['rate_limit_wait_seconds']:.1f}s waiting on rate limits, "
                   f"{t['api_call_seconds']:.1f}s inside API calls "
                   f"(thread-seconds, summed across {self.config_snapshot.get('max_concurrency', '?')} workers)")
        out.append(f"Wall clock      : {t['run_wall_seconds']:.1f}s "
                   f"({t['wait_threadsec_per_100s_wall']:.0f} wait thread-seconds per 100s of wall clock)")
        for n in self.notes:
            out.append(f"NOTE: {n}")
        out.append("=" * len(sep))
        return "\n".join(out)

    def print_summary(self):
        print(self.format_summary())
