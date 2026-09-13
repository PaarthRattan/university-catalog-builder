import requests
import asyncio
import aiohttp
import time
from typing import List, Dict, Optional
import logging
from config import (WIKIPEDIA_API_URL, WIKIPEDIA_USER_AGENT, WIKIPEDIA_REQUESTS_PER_SECOND,
                    WIKIPEDIA_MAX_CONCURRENCY, BATCH_SIZE, EXTRACT_BATCH_SIZE,
                    MAX_RETRIES, RETRY_DELAY)

logger = logging.getLogger(__name__)

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

            params = {
                'action': 'query',
                'pageids': '|'.join(map(str, batch_ids)),
                'prop': 'extracts|info|categories',
                'exintro': True,
                'explaintext': True,
                'exlimit': EXTRACT_BATCH_SIZE,
                'inprop': 'url',
                'format': 'json'
            }
            
            data = self._make_request(params)
            if not data or 'query' not in data:
                continue
            
            pages = data['query'].get('pages', {})
            for page_id, page_info in pages.items():
                if 'extract' in page_info:
                    categories = []
                    if 'categories' in page_info:
                        categories = [cat['title'] for cat in page_info['categories']]
                    
                    page_data[int(page_id)] = {
                        'title': page_info.get('title', ''),
                        'extract': page_info.get('extract', ''),
                        'url': page_info.get('fullurl', ''),
                        'categories': categories
                    }
        
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
        """Async version of get_page_extracts for better performance"""
        page_data = {}

        # Bound in-flight requests with a semaphore. A bare asyncio.gather over
        # every batch opens one connection per batch simultaneously, which for a
        # large page set means thousands of concurrent requests at Wikipedia --
        # both impolite and a reliable way to get throttled or blocked.
        semaphore = asyncio.Semaphore(max(1, WIKIPEDIA_MAX_CONCURRENCY))

        async with aiohttp.ClientSession() as session:
            async def bounded(batch_ids):
                async with semaphore:
                    return await self._async_get_batch(session, batch_ids)

            tasks = [bounded(page_ids[i:i + EXTRACT_BATCH_SIZE])
                     for i in range(0, len(page_ids), EXTRACT_BATCH_SIZE)]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            for result in results:
                if isinstance(result, dict):
                    page_data.update(result)
                elif isinstance(result, Exception):
                    logger.error(f"Async request failed: {result}")
        
        return page_data
    
    async def _async_get_batch(self, session: aiohttp.ClientSession, page_ids: List[int]) -> Dict[int, Dict]:
        """Async helper to get a batch of page extracts"""
        params = {
            'action': 'query',
            'pageids': '|'.join(map(str, page_ids)),
            'prop': 'extracts|info|categories',
            'exintro': True,
            'explaintext': True,
            'exlimit': EXTRACT_BATCH_SIZE,
            'inprop': 'url',
            'format': 'json'
        }

        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            async with session.get(self.api_url, params=params, headers=self.headers) as response:
                data = await response.json()
                
                page_data = {}
                if 'query' in data:
                    pages = data['query'].get('pages', {})
                    for page_id, page_info in pages.items():
                        if 'extract' in page_info:
                            categories = []
                            if 'categories' in page_info:
                                categories = [cat['title'] for cat in page_info['categories']]
                            
                            page_data[int(page_id)] = {
                                'title': page_info.get('title', ''),
                                'extract': page_info.get('extract', ''),
                                'url': page_info.get('fullurl', ''),
                                'categories': categories
                            }
                
                return page_data
        
        except Exception as e:
            logger.error(f"Async batch request failed: {e}")
            return {}