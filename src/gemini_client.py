"""Gemini client with enforced RPM/TPM/RPD limits, honest error handling and
optional batching.

Design notes
------------
* Every call goes through ``_generate``, which is the single place that reserves
  rate-limit budget, retries, and records metrics.
* A failed call raises ``GeminiCallError``. It never returns a plausible-looking
  default. The previous implementation returned ``{"is_university": False}`` on
  API errors, which silently converted every rate-limit rejection into a wrong
  negative classification; callers could not tell a real "no" from a failure.
* Retries are applied only to errors classified as transient. A malformed
  request or a bad API key fails immediately instead of burning the retry budget.
"""

import json
import logging
import random
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_RPM, GEMINI_TPM, GEMINI_RPD,
    GEMINI_MAX_RETRIES, GEMINI_BACKOFF_BASE, GEMINI_BACKOFF_FACTOR,
    GEMINI_BACKOFF_MAX, GEMINI_BACKOFF_JITTER, GEMINI_MAX_CONCURRENCY,
)
from .rate_limiter import GeminiRateLimiter, DailyQuotaExhausted, RateLimitExceeded

logger = logging.getLogger(__name__)

# Rough chars-per-token for the pre-call TPM reservation. The reservation is
# reconciled against the API's reported usage immediately after each call, so
# this only needs to be in the right ballpark.
CHARS_PER_TOKEN = 4


class GeminiCallError(Exception):
    """A Gemini call failed after exhausting retries."""

    def __init__(self, message: str, retries: int = 0, rate_limit_hits: int = 0,
                 wait_seconds: float = 0.0):
        super().__init__(message)
        self.retries = retries
        self.rate_limit_hits = rate_limit_hits
        self.wait_seconds = wait_seconds


def _status_code(exc: Exception) -> Optional[int]:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    for attr in ("status_code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    m = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(m.group(1)) if m else None


def is_rate_limit_error(exc: Exception) -> bool:
    """True for a server-side 429 / RESOURCE_EXHAUSTED."""
    if _status_code(exc) == 429:
        return True
    text = f"{getattr(exc, 'status', '')} {exc}".upper()
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def is_retryable_error(exc: Exception) -> bool:
    """True for transient failures worth another attempt.

    Deliberately excludes 400/401/403/404 -- a malformed request, a bad key or a
    missing model will fail identically on every retry, and retrying them hides
    the real cause behind a generic 'failed after N attempts'.
    """
    if is_rate_limit_error(exc):
        return True
    code = _status_code(exc)
    if code is not None:
        return code in (408, 409, 500, 502, 503, 504)
    return isinstance(exc, (TimeoutError, ConnectionError))


def parse_retry_after(exc: Exception) -> Optional[float]:
    """Extract the server's requested retry delay, if it gave one.

    Gemini supplies this two ways: a google.rpc.RetryInfo entry in error details,
    and a human-readable 'Please retry in 39.97s' in the message.
    """
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        for item in details.get("error", {}).get("details", []) or []:
            if str(item.get("@type", "")).endswith("RetryInfo"):
                delay = item.get("retryDelay")
                if isinstance(delay, str):
                    m = re.match(r"^([\d.]+)s?$", delay.strip())
                    if m:
                        return float(m.group(1))
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.IGNORECASE)
    return float(m.group(1)) if m else None


def extract_quota_violations(exc: Exception) -> List[Tuple[str, str]]:
    """Pull (quotaId, quotaValue) pairs out of a 429 so we can learn real limits."""
    out = []
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        for item in details.get("error", {}).get("details", []) or []:
            if str(item.get("@type", "")).endswith("QuotaFailure"):
                for v in item.get("violations", []) or []:
                    qid = v.get("quotaId")
                    qval = v.get("quotaValue")
                    if qid and qval:
                        out.append((str(qid), str(qval)))
    return out


def is_daily_quota_violation(exc: Exception) -> bool:
    """True when a 429 names a PER-DAY quota, as opposed to a per-minute one."""
    for qid, _ in extract_quota_violations(exc):
        if "perday" in qid.lower():
            return True
    return bool(re.search(r"per\s*day|daily limit", str(exc), re.IGNORECASE))


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _coerce_int(value) -> Optional[int]:
    """Accept 1887, '1887', '1887 AD'; reject anything without a clean integer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        m = re.search(r"-?\d+", value.replace(",", ""))
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                return None
    return None


class GeminiClient:
    def __init__(self, metrics=None, rate_limiter=None, state_path: str = None,
                 model: str = None):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = model or GEMINI_MODEL
        self.metrics = metrics
        self.limiter = rate_limiter or GeminiRateLimiter(
            rpm=GEMINI_RPM, tpm=GEMINI_TPM, rpd=GEMINI_RPD, state_path=state_path)
        # Never hold more requests in flight than the per-minute budget can retire.
        self.max_concurrency = max(1, min(GEMINI_MAX_CONCURRENCY, GEMINI_RPM))
        self._semaphore = threading.Semaphore(self.max_concurrency)
        self._config = types.GenerateContentConfig(
            # 'minimal' is the current-model equivalent of the old
            # thinking_budget=0: no thinking tokens spent on a structured
            # extraction task that does not benefit from them.
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            response_mime_type="application/json",
        )

    # ------------------------------------------------------------- core call

    def _generate(self, prompt: str, purpose: str) -> Dict:
        """Run one Gemini request with full rate-limit and retry handling.

        Returns the parsed JSON object. Raises GeminiCallError on failure and
        DailyQuotaExhausted when the daily budget is gone.
        """
        estimated = max(1, len(prompt) // CHARS_PER_TOKEN)
        retries = 0
        rl_hits = 0
        waited_total = 0.0
        delay = GEMINI_BACKOFF_BASE
        last_exc = None

        with self._semaphore:
            for attempt in range(GEMINI_MAX_RETRIES + 1):
                # DailyQuotaExhausted intentionally propagates: the caller must
                # checkpoint and stop, not sleep for hours.
                wait = self.limiter.acquire(estimated)
                waited_total += wait
                throttled = wait > 0

                t0 = time.time()
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name, contents=prompt, config=self._config)
                    api_seconds = time.time() - t0

                    usage = getattr(response, "usage_metadata", None)
                    in_tok = getattr(usage, "prompt_token_count", 0) or 0
                    out_tok = getattr(usage, "candidates_token_count", 0) or 0
                    think_tok = getattr(usage, "thoughts_token_count", 0) or 0
                    self.limiter.record_actual_tokens(estimated, in_tok + out_tok + think_tok)

                    text = response.text
                    if not text:
                        raise ValueError("empty response body")
                    parsed = json.loads(_strip_code_fence(text))
                    if not isinstance(parsed, dict):
                        raise ValueError(f"expected JSON object, got {type(parsed).__name__}")

                    if self.metrics:
                        self.metrics.record_request(
                            input_tokens=in_tok, output_tokens=out_tok,
                            thinking_tokens=think_tok, api_seconds=api_seconds,
                            wait_seconds=waited_total, retries=retries,
                            rate_limit_hits=rl_hits, throttled=throttled)
                    return parsed

                except (json.JSONDecodeError, ValueError) as exc:
                    # Malformed model output. The request itself succeeded and
                    # was billed, so the slot stays spent.
                    last_exc = exc
                    if attempt >= GEMINI_MAX_RETRIES:
                        break
                    retries += 1
                    logger.warning("%s: unparseable response (attempt %d/%d): %s",
                                   purpose, attempt + 1, GEMINI_MAX_RETRIES, exc)
                    slept = self._sleep_backoff(delay)
                    waited_total += slept
                    delay = min(delay * GEMINI_BACKOFF_FACTOR, GEMINI_BACKOFF_MAX)

                except Exception as exc:
                    last_exc = exc
                    # The request was rejected, so give the reserved slot back
                    # rather than counting a rejection against our own budget.
                    self.limiter.release_slot()

                    if is_rate_limit_error(exc):
                        rl_hits += 1
                        violations = extract_quota_violations(exc)
                        for qid, qval in violations:
                            self.limiter.note_server_quota(qid, qval)
                        # Always log which quota the server actually named --
                        # "rate limited" without the quota id is undiagnosable.
                        logger.info("%s: 429 quota violations: %s",
                                    purpose, violations or "(none reported)")
                        if is_daily_quota_violation(exc):
                            snap = self.limiter.snapshot()
                            daily = DailyQuotaExhausted(
                                snap["requests_last_day"], snap["rpd_limit"], 0.0)
                            daily.violations = violations
                            daily.server_message = str(exc)[:500]
                            raise daily from exc
                        retry_after = parse_retry_after(exc)
                        if attempt >= GEMINI_MAX_RETRIES:
                            break
                        retries += 1
                        sleep_for = retry_after if retry_after is not None else delay
                        logger.warning("%s: 429 from server, backing off %.1fs "
                                       "(attempt %d/%d)%s", purpose, sleep_for,
                                       attempt + 1, GEMINI_MAX_RETRIES,
                                       " [server Retry-After]" if retry_after else "")
                        slept = self._sleep_backoff(sleep_for, honor_exactly=retry_after is not None)
                        waited_total += slept
                        delay = min(delay * GEMINI_BACKOFF_FACTOR, GEMINI_BACKOFF_MAX)
                        continue

                    if not is_retryable_error(exc):
                        # Fail fast and loudly on 400/401/403/404 and friends.
                        logger.error("%s: non-retryable error: %s", purpose, exc)
                        if self.metrics:
                            self.metrics.record_failure(retries=retries,
                                                        rate_limit_hits=rl_hits,
                                                        wait_seconds=waited_total)
                        raise GeminiCallError(f"{purpose}: {exc}", retries, rl_hits,
                                              waited_total) from exc

                    if attempt >= GEMINI_MAX_RETRIES:
                        break
                    retries += 1
                    logger.warning("%s: transient error (attempt %d/%d): %s",
                                   purpose, attempt + 1, GEMINI_MAX_RETRIES, exc)
                    slept = self._sleep_backoff(delay)
                    waited_total += slept
                    delay = min(delay * GEMINI_BACKOFF_FACTOR, GEMINI_BACKOFF_MAX)

        if self.metrics:
            self.metrics.record_failure(retries=retries, rate_limit_hits=rl_hits,
                                        wait_seconds=waited_total)
        raise GeminiCallError(
            f"{purpose}: failed after {retries} retries: {last_exc}",
            retries, rl_hits, waited_total)

    def _sleep_backoff(self, base: float, honor_exactly: bool = False) -> float:
        """Sleep with full jitter. Returns seconds actually slept.

        When the server supplied a Retry-After we wait at least that long and add
        only positive jitter -- undershooting a server-specified delay just earns
        another 429.
        """
        if honor_exactly:
            delay = base + random.uniform(0, GEMINI_BACKOFF_JITTER * base)
        else:
            jitter = GEMINI_BACKOFF_JITTER * base
            delay = max(0.0, base + random.uniform(-jitter, jitter))
        delay = min(delay, GEMINI_BACKOFF_MAX)
        time.sleep(delay)
        return delay

    # -------------------------------------------------------- classification

    def is_university(self, title: str, extract: str,
                      categories: List[str] = None) -> Dict:
        """Classify a single page. Raises GeminiCallError on failure."""
        categories_text = ", ".join(categories) if categories else "None"
        prompt = f"""Analyze this Wikipedia page to determine if it represents a university or higher education institution.

Title: {title}
Extract: {extract[:1000]}
Categories: {categories_text}

Consider these criteria:
- Universities, colleges, and higher education institutions
- Institutions that grant degrees (bachelor's, master's, PhD)
- Exclude: high schools, elementary schools, research institutes without degree programs, academic departments within universities

Respond with valid JSON only:
{{"is_university": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""
        result = self._generate(prompt, f"classify({title})")
        if "is_university" not in result or "confidence" not in result:
            raise GeminiCallError(f"classify({title}): missing required fields in {result}")
        return {
            "is_university": bool(result["is_university"]),
            "confidence": float(result.get("confidence") or 0.0),
            "reasoning": result.get("reasoning", ""),
        }

    def batch_university_classification(self, pages: List[Dict],
                                        batch_size: int = None) -> List[Dict]:
        """Classify a batch of pages in ONE request.

        Unlike the previous implementation this does not silently truncate the
        input: every page passed in is classified, chunked across as many
        requests as the batch size requires. Pages the model omits from its
        response are reported with ``ok=False`` so the caller can retry or park
        them rather than mistaking silence for a negative classification.
        """
        if not pages:
            return []
        size = batch_size or len(pages)
        size = max(1, size)

        results: List[Dict] = []
        for start in range(0, len(pages), size):
            chunk = pages[start:start + size]
            results.extend(self._classify_chunk(chunk))
        return results

    def _classify_chunk(self, pages: List[Dict]) -> List[Dict]:
        batch_data = [{
            "id": i,
            "title": p["title"],
            "extract": (p.get("extract") or "")[:500],
            "categories": ", ".join(p.get("categories", []) or []) if isinstance(
                p.get("categories"), list) else (p.get("categories") or "None"),
        } for i, p in enumerate(pages)]

        prompt = f"""Classify each of the following Wikipedia pages to determine if they represent universities or higher education institutions.

Pages to analyze:
{json.dumps(batch_data, indent=2)}

For each page, consider:
- Universities, colleges, and higher education institutions
- Institutions that grant degrees (bachelor's, master's, PhD)
- Exclude: high schools, elementary schools, research institutes without degree programs

You MUST return exactly one result for every id from 0 to {len(pages) - 1}.

Respond with valid JSON only:
{{"results": [{{"id": 0, "is_university": true/false, "confidence": 0.0-1.0}}]}}
"""
        try:
            parsed = self._generate(prompt, f"classify_batch(n={len(pages)})")
        except GeminiCallError as exc:
            logger.error("Batch classification failed for %d pages: %s", len(pages), exc)
            return [{"page": p, "ok": False, "error": str(exc)} for p in pages]

        by_id = {}
        for item in parsed.get("results", []) or []:
            idx = _coerce_int(item.get("id"))
            if idx is not None and 0 <= idx < len(pages) and idx not in by_id:
                by_id[idx] = item

        out = []
        for i, page in enumerate(pages):
            item = by_id.get(i)
            if item is None:
                # Explicitly surfaced rather than defaulted to False.
                out.append({"page": page, "ok": False,
                            "error": "model omitted this id from its response"})
                continue
            out.append({
                "page": page,
                "ok": True,
                "is_university": bool(item.get("is_university", False)),
                "confidence": float(item.get("confidence") or 0.0),
            })
        missing = len(pages) - len(by_id)
        if missing:
            logger.warning("Batch of %d: model omitted %d ids", len(pages), missing)
        return out

    # ------------------------------------------------------------ extraction

    _EXTRACT_SCHEMA = """{
  "name": "official university name",
  "city": "city name or null",
  "country": "country name or null",
  "founded": founding year as integer or null,
  "closed": closure year as integer or null,
  "type": "public/private/other or null",
  "student_count": approximate number of students as integer or null,
  "notable_programs": ["list of notable academic programs"],
  "coordinates": {"lat": number, "lng": number} or null
}"""

    _EXTRACT_GUIDELINES = """Guidelines:
- Use exact names and spellings from the content
- For years, extract only the numeric year (e.g. 1887, not "1887 AD")
- If information is unclear or not mentioned, use null
- Be conservative: only include information you are confident about"""

    def extract_university_data(self, title: str, extract: str,
                                full_content: str = None) -> Dict:
        """Extract structured fields for one university."""
        content = full_content if full_content else extract
        prompt = f"""Extract detailed information about this university from the Wikipedia content.

Title: {title}
Content: {(content or '')[:2000]}

Extract the following information and respond with valid JSON only:
{self._EXTRACT_SCHEMA}

{self._EXTRACT_GUIDELINES}
"""
        result = self._generate(prompt, f"extract({title})")
        return self._clean_extraction(result, title)

    @staticmethod
    def _clean_extraction(result: Dict, fallback_title: str) -> Dict:
        coords = result.get("coordinates")
        if isinstance(coords, dict):
            lat, lng = coords.get("lat"), coords.get("lng")
            coords = ({"lat": float(lat), "lng": float(lng)}
                      if isinstance(lat, (int, float)) and isinstance(lng, (int, float))
                      else None)
        else:
            coords = None
        programs = result.get("notable_programs")
        return {
            "name": result.get("name") or fallback_title,
            "city": result.get("city"),
            "country": result.get("country"),
            "founded": _coerce_int(result.get("founded")),
            "closed": _coerce_int(result.get("closed")),
            "type": result.get("type"),
            "student_count": _coerce_int(result.get("student_count")),
            "notable_programs": programs if isinstance(programs, list) else [],
            "coordinates": coords,
        }

    # ------------------------------------------- combined classify + extract

    def classify_and_extract(self, pages: List[Dict],
                             batch_size: int = None) -> List[Dict]:
        """One Gemini call that both classifies AND extracts, for a batch of pages.

        This is the merged path behind GEMINI_COMBINED_FILTER_EXTRACT. It halves
        the request count for pages that turn out to be universities, at the cost
        of asking the model to do two jobs at once -- hence the A/B flag.
        """
        if not pages:
            return []
        size = max(1, batch_size or len(pages))
        out: List[Dict] = []
        for start in range(0, len(pages), size):
            out.extend(self._combined_chunk(pages[start:start + size]))
        return out

    def _combined_chunk(self, pages: List[Dict]) -> List[Dict]:
        batch_data = [{
            "id": i,
            "title": p["title"],
            "content": (p.get("extract") or "")[:2000],
        } for i, p in enumerate(pages)]

        prompt = f"""For each Wikipedia page below, do TWO things:
1. Decide whether it represents a university or higher education institution.
   - Include universities, colleges and degree-granting higher education institutions
   - Exclude high schools, elementary schools, research institutes without degree
     programs, and academic departments within universities
2. If and only if it IS a university, extract its structured details.
   If it is NOT a university, set "details" to null.

Pages:
{json.dumps(batch_data, indent=2)}

{self._EXTRACT_GUIDELINES}

You MUST return exactly one result for every id from 0 to {len(pages) - 1}.

Respond with valid JSON only:
{{"results": [{{"id": 0, "is_university": true/false, "confidence": 0.0-1.0,
  "details": {self._EXTRACT_SCHEMA} }}]}}
"""
        try:
            parsed = self._generate(prompt, f"combined(n={len(pages)})")
        except GeminiCallError as exc:
            logger.error("Combined classify+extract failed for %d pages: %s", len(pages), exc)
            return [{"page": p, "ok": False, "error": str(exc)} for p in pages]

        by_id = {}
        for item in parsed.get("results", []) or []:
            idx = _coerce_int(item.get("id"))
            if idx is not None and 0 <= idx < len(pages) and idx not in by_id:
                by_id[idx] = item

        out = []
        for i, page in enumerate(pages):
            item = by_id.get(i)
            if item is None:
                out.append({"page": page, "ok": False,
                            "error": "model omitted this id from its response"})
                continue
            is_uni = bool(item.get("is_university", False))
            details = item.get("details")
            out.append({
                "page": page,
                "ok": True,
                "is_university": is_uni,
                "confidence": float(item.get("confidence") or 0.0),
                "details": (self._clean_extraction(details, page["title"])
                            if is_uni and isinstance(details, dict) else None),
            })
        return out

    def limiter_snapshot(self) -> Dict:
        return self.limiter.snapshot()
