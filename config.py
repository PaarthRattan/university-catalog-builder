import os
from dotenv import load_dotenv

load_dotenv()

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}")

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')
WIKIPEDIA_USER_AGENT = os.getenv('WIKIPEDIA_USER_AGENT', 'UniversityCatalog/1.0')
WIKIPEDIA_API_URL = 'https://en.wikipedia.org/w/api.php'

# Database Configuration
DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/universities.db')

# Rate Limiting -- Wikipedia
WIKIPEDIA_REQUESTS_PER_SECOND = _env_int('WIKIPEDIA_REQUESTS_PER_SECOND', 10)
WIKIPEDIA_MAX_CONCURRENCY = _env_int('WIKIPEDIA_MAX_CONCURRENCY', 8)
# Use the concurrent aiohttp collection path. Set 0 to fall back to the serial
# requests path (both produce identical output; see _parse_extract_payload).
WIKIPEDIA_USE_ASYNC = _env_bool('WIKIPEDIA_USE_ASYNC', True)

# Rate Limiting -- Gemini
#
# These are the THREE limits the client enforces locally. Exceeding any one of
# them is treated as a 429 before the request leaves this process.
#
# RPM below is MEASURED, not assumed: on 2026-09-13 the live API rejected a
# burst against gemini-3.6-flash with
#   quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier
#   quotaValue: "5"
# Re-check any time with `python main.py probe-limits`.
#
# RPD is also MEASURED, from a 429 on the same day:
#   quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier
#   quotaValue: "20"
# Twenty requests per day is the binding constraint on this tier by a wide
# margin -- it is what makes request batching essential rather than merely nice.
#
# TPM remains unverified: at 5 RPM the token limit is unreachable in practice
# (five requests of even 10k tokens is 50k/min), so the API has never reported
# it. The client learns the real value from the first 429 that names it.
GEMINI_RPM = _env_int('GEMINI_RPM', 5)            # requests per minute (MEASURED)
GEMINI_TPM = _env_int('GEMINI_TPM', 250_000)      # tokens per minute (unverified)
GEMINI_RPD = _env_int('GEMINI_RPD', 20)           # requests per day (MEASURED)

# Retry / backoff. Applies to transient failures only; see src/gemini_client.py
# for which errors are classified retryable.
GEMINI_MAX_RETRIES = _env_int('GEMINI_MAX_RETRIES', 5)
GEMINI_BACKOFF_BASE = _env_float('GEMINI_BACKOFF_BASE', 1.0)     # seconds
GEMINI_BACKOFF_FACTOR = _env_float('GEMINI_BACKOFF_FACTOR', 2.0)
GEMINI_BACKOFF_MAX = _env_float('GEMINI_BACKOFF_MAX', 60.0)      # seconds
GEMINI_BACKOFF_JITTER = _env_float('GEMINI_BACKOFF_JITTER', 0.3) # +/- fraction

# Throughput
# Pages packed into a single classification request. 1 disables batching.
GEMINI_CLASSIFY_BATCH_SIZE = _env_int('GEMINI_CLASSIFY_BATCH_SIZE', 10)
# Pages packed into a single combined classify+extract request.
GEMINI_COMBINED_BATCH_SIZE = _env_int('GEMINI_COMBINED_BATCH_SIZE', 5)
# A/B flag: when true, filtering and extraction collapse into ONE Gemini call
# per page instead of two separate passes. See docs in README.
GEMINI_COMBINED_FILTER_EXTRACT = _env_bool('GEMINI_COMBINED_FILTER_EXTRACT', False)
# In-flight Gemini requests. Capped at GEMINI_RPM by the client -- no point
# holding more sockets open than the per-minute budget can retire.
GEMINI_MAX_CONCURRENCY = _env_int('GEMINI_MAX_CONCURRENCY', 8)

# Cost estimation (USD per 1M tokens). Used only for the estimated-cost line in
# the metrics report. Set to your tier's real prices; 0.0 means "free tier, do
# not estimate a dollar figure".
GEMINI_PRICE_INPUT_PER_MTOK = _env_float('GEMINI_PRICE_INPUT_PER_MTOK', 0.0)
GEMINI_PRICE_OUTPUT_PER_MTOK = _env_float('GEMINI_PRICE_OUTPUT_PER_MTOK', 0.0)

# Classification confidence required to treat a page as a university. Used
# consistently by both the filter stage and the extract stage.
UNIVERSITY_CONFIDENCE_THRESHOLD = _env_float('UNIVERSITY_CONFIDENCE_THRESHOLD', 0.5)

# Give up on a page after this many failed Gemini attempts, so a page that the
# model never returns a result for cannot spin the filter loop forever.
MAX_PAGE_ATTEMPTS = _env_int('MAX_PAGE_ATTEMPTS', 3)

# Processing Configuration
# NOTE: BATCH_SIZE controls WIKIPEDIA API chunking only (pageids per query).
# It has never had any effect on Gemini batching -- see GEMINI_*_BATCH_SIZE.
BATCH_SIZE = _env_int('BATCH_SIZE', 50)
# Wikipedia caps `exlimit` at 20 when requesting extracts. Sending BATCH_SIZE=50
# pageids silently returns extracts for only the first 20, leaving the rest with
# empty content -- which then gets classified by Gemini on no information at all.
EXTRACT_BATCH_SIZE = _env_int('EXTRACT_BATCH_SIZE', 20)
MAX_RETRIES = _env_int('MAX_RETRIES', 3)      # Wikipedia client retries
RETRY_DELAY = _env_float('RETRY_DELAY', 1.0)  # Wikipedia client base delay

# Metrics
METRICS_DIR = os.getenv('METRICS_DIR', 'metrics')

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# University Categories to Search
UNIVERSITY_CATEGORIES = [
    'Category:Universities by country',
    'Category:Educational institutions by country',
    'Category:Universities and colleges',
    'Category:Higher education institutions',
    'Category:Public universities',
    'Category:Private universities',
    'Category:Technical universities',
    'Category:Medical schools',
    'Category:Business schools',
    'Category:Art schools'
]

# A small, fixed, reproducible subset used for before/after benchmark runs.
BENCHMARK_CATEGORIES = [
    'Category:Universities and colleges in Delaware',
    'Category:Universities and colleges in Rhode Island'
]
