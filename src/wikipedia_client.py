import requests
import asyncio
import aiohttp
import time
from typing import List, Dict, Optional
import logging
import random
from config import (WIKIPEDIA_API_URL, WIKIPEDIA_USER_AGENT, WIKIPEDIA_REQUESTS_PER_SECOND,
                    WIKIPEDIA_MAX_CONCURRENCY, BATCH_SIZE, EXTRACT_BATCH_SIZE,
                    MAX_RETRIES, RETRY_DELAY)

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Global request spacing across concurrent coroutines.

    The obvious approach -- `await asyncio.sleep(delay)` at the top of each task
    -- does not rate limit anything. It delays each task by a fixed amount
    individually, so N concurrent tasks still issue N requests per `delay`
    window: with 8 workers and a 0.1s delay that is ~80 req/s against a
    configured ceiling of 10.

    This reserves a slot on a shared timeline instead, so the aggregate rate
    holds no matter how many coroutines are running. The reservation is made
    under the lock and the sleep happens outside it, so waiting tasks do not
    serialise behind each other.
    """

    def __init__(self, rate_per_second: float):
        self._min_interval = 1.0 / max(rate_per_second, 0.001)
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        async with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._min_interval
            wait = slot - now
        if wait > 0:
            await asyncio.sleep(wait)
        return wait

class WikipediaClient:
    def __init__(self):
        self.api_url = WIKIPEDIA_API_URL
        self.headers = {'User-Agent': WIKIPEDIA_USER_AGENT}
        self.last_request_time = 0
        self.rate_limit_delay = 1.0 / WIKIPEDIA_REQUESTS_PER_SECOND
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict, retries: int = 0) -> Optional[Dict]:
        """Make a request to Wikipedia API with retry logic"""
        self._rate_limit()
        
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if retries < MAX_RETRIES:
                logger.warning(f"Request failed, retrying ({retries + 1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY * (retries + 1))
                return self._make_request(params, retries + 1)
            else:
                logger.error(f"Request failed after {MAX_RETRIES} retries: {e}")
                return None
    
    def get_category_members(self, category: str, limit: int = 500) -> List[Dict]:
        """Get article pages in a category.

        Restricted to namespace 0 (articles). Without this the listing also
        returns the category's own subcategories as if they were pages, which
        then get stored with no extract and sent to Gemini for classification --
        spending quota to be told that "Category:Arts schools in Delaware" is not
        a university. Subcategories are enumerated separately by
        get_subcategories().
        """
        pages = []
        continue_token = None

        while True:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': category,
                'cmlimit': min(limit, 500),
                'cmnamespace': 0,
                'format': 'json'
            }
            
            if continue_token:
                params['cmcontinue'] = continue_token
            
            data = self._make_request(params)
            if not data or 'query' not in data:
                break
            
            category_members = data['query'].get('categorymembers', [])
            for member in category_members:
                pages.append({
                    'page_id': member['pageid'],
                    'title': member['title'],
                    'url': f"https://en.wikipedia.org/wiki/{member['title'].replace(' ', '_')}"
                })
            
            # Check if there are more pages
            if 'continue' in data and len(category_members) == 500:
                continue_token = data['continue']['cmcontinue']
                if len(pages) >= limit:
                    break
            else:
                break
        
        logger.info(f"Found {len(pages)} pages in category {category}")
        return pages[:limit]
    
    def get_page_extracts(self, page_ids: List[int]) -> Dict[int, Dict]:
        """Get page extracts and basic info for multiple pages"""
        page_data = {}
        
        # Chunked at EXTRACT_BATCH_SIZE, not BATCH_SIZE: the extracts API returns
        # content for at most `exlimit` (max 20) pages per request.
        for i in range(0, len(page_ids), EXTRACT_BATCH_SIZE):
            batch_ids = page_ids[i:i + EXTRACT_BATCH_SIZE]
            base = self._extract_params(batch_ids)
            cont = {}
            for _ in range(20):
                data = self._make_request({**base, **cont})
                if not data or 'query' not in data:
                    break
                self._merge_extract_payload(page_data, data)
                if 'continue' not in data:
                    break
                cont = data['continue']

        logger.info(f"Retrieved extracts for {len(page_data)} pages")
        return page_data
    
    def search_universities(self, query: str, limit: int = 100) -> List[Dict]:
        """Search for university-related pages"""
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': f'{query} university OR college OR "higher education"',
            'srlimit': limit,
            'srnamespace': 0,  # Main namespace only
            'format': 'json'
        }
        
        data = self._make_request(params)
        if not data or 'query' not in data:
            return []
        
        search_results = data['query'].get('search', [])
        pages = []
        
        for result in search_results:
            pages.append({
                'page_id': result['pageid'],
                'title': result['title'],
                'url': f"https://en.wikipedia.org/wiki/{result['title'].replace(' ', '_')}",
                'snippet': result.get('snippet', '')
            })
        
        logger.info(f"Found {len(pages)} pages for search: {query}")
        return pages
    
    def get_subcategories(self, category: str) -> List[str]:
        """Get subcategories of a given category"""
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': category,
            'cmtype': 'subcat',
            'cmlimit': 500,
            'format': 'json'
        }
        
        data = self._make_request(params)
        if not data or 'query' not in data:
            return []
        
        subcategories = [member['title'] for member in data['query'].get('categorymembers', [])]
        logger.info(f"Found {len(subcategories)} subcategories in {category}")
        return subcategories
    
    async def async_get_page_extracts(self, page_ids: List[int]) -> Dict[int, Dict]:
        """Fetch extracts for many pages concurrently.

        Bounded by a semaphore (connection count) AND a shared rate limiter
        (requests per second). Both are needed: the semaphore caps how many
        sockets are open at once, the limiter caps how fast requests leave.
        """
        if not page_ids:
            return {}

        page_data: Dict[int, Dict] = {}
        semaphore = asyncio.Semaphore(max(1, WIKIPEDIA_MAX_CONCURRENCY))
        limiter = AsyncRateLimiter(WIKIPEDIA_REQUESTS_PER_SECOND)

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(headers=self.headers, timeout=timeout) as session:
            async def bounded(batch_ids):
                async with semaphore:
                    return await self._async_get_batch(session, batch_ids, limiter)

            tasks = [bounded(page_ids[i:i + EXTRACT_BATCH_SIZE])
                     for i in range(0, len(page_ids), EXTRACT_BATCH_SIZE)]

            failures = 0
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, dict):
                    page_data.update(result)
                else:
                    failures += 1
                    logger.error("Async batch failed: %s", result)

        if failures:
            logger.warning("%d of %d async batches failed", failures, len(tasks))
        logger.info("Retrieved extracts for %d pages (async)", len(page_data))
        return page_data

    async def _async_get_batch(self, session: aiohttp.ClientSession,
                               page_ids: List[int],
                               limiter: "AsyncRateLimiter") -> Dict[int, Dict]:
        """Fetch one batch, retrying transient failures with backoff."""
        base = self._extract_params(page_ids)
        page_data: Dict[int, Dict] = {}
        cont: Dict = {}

        # Follow `continue` until the API stops offering one, so paginated
        # categories are collected in full. Bounded to keep a pathological
        # response from looping forever.
        for _ in range(20):
            params = {**base, **cont}
            data = await self._async_request(session, params, limiter)
            if data is None:
                break
            self._merge_extract_payload(page_data, data)
            if 'continue' not in data:
                break
            cont = data['continue']
        return page_data

    async def _async_request(self, session, params: Dict,
                             limiter: "AsyncRateLimiter") -> Optional[Dict]:
        """One rate-limited request with retry and exponential backoff."""
        for attempt in range(MAX_RETRIES + 1):
            await limiter.acquire()
            try:
                async with session.get(self.api_url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as exc:
                if attempt >= MAX_RETRIES:
                    logger.error("Async request failed after %d retries: %s",
                                 MAX_RETRIES, exc)
                    raise
                delay = RETRY_DELAY * (2 ** attempt)
                delay += random.uniform(0, 0.3 * delay)
                logger.warning("Async retry %d/%d in %.1fs: %s",
                               attempt + 1, MAX_RETRIES, delay, exc)
                await asyncio.sleep(delay)
        return None

    @staticmethod
    def _merge_extract_payload(page_data: Dict[int, Dict], data: Dict) -> None:
        """Accumulate one API response into page_data, in place.

        Merges rather than replaces, because `prop=categories` is paginated:
        MediaWiki applies `cllimit` across the WHOLE query, not per page, so a
        20-page request returns categories for only the first page or two and
        signals the rest via a `continue` token. Neither path followed that
        token, so ~90% of pages were stored with an empty category list -- and
        the classification prompt feeds those categories to the model. Pages
        were being classified with their strongest signal missing.
        """
        for page_id, info in data.get('query', {}).get('pages', {}).items():
            pid = int(page_id)
            entry = page_data.get(pid)
            if entry is None:
                if 'extract' not in info:
                    continue
                entry = {
                    'title': info.get('title', ''),
                    'extract': info.get('extract', ''),
                    'url': info.get('fullurl', ''),
                    'categories': [],
                }
                page_data[pid] = entry
            seen = set(entry['categories'])
            for cat in info.get('categories', []) or []:
                title = cat.get('title')
                if title and title not in seen:
                    seen.add(title)
                    entry['categories'].append(title)

    @classmethod
    def _extract_params(cls, page_ids: List[int]) -> Dict:
        """Query params shared by both paths, so they cannot drift apart."""
        return {
            'action': 'query',
            'pageids': '|'.join(map(str, page_ids)),
            'prop': 'extracts|info|categories',
            'exintro': 1,
            'explaintext': 1,
            'exlimit': EXTRACT_BATCH_SIZE,
            'cllimit': 'max',
            'clshow': '!hidden',
            'inprop': 'url',
            'format': 'json',
        }
