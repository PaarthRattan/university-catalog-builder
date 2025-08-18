from google import genai
from google.genai import types
import json
import time
import asyncio
from typing import Dict, List, Optional
import logging
from config import GEMINI_API_KEY, GEMINI_REQUESTS_PER_MINUTE, MAX_RETRIES, RETRY_DELAY

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = 'gemini-2.5-flash'
        self.rate_limit_delay = 60.0 / GEMINI_REQUESTS_PER_MINUTE
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def is_university(self, title: str, extract: str, categories: List[str] = None) -> Dict:
        """Determine if a Wikipedia page represents a university"""
        self._rate_limit()
        
        categories_text = ", ".join(categories) if categories else "None"
        
        prompt = f"""
        Analyze this Wikipedia page to determine if it represents a university or higher education institution.

        Title: {title}
        Extract: {extract[:1000]}...
        Categories: {categories_text}

        Consider these criteria:
        - Universities, colleges, and higher education institutions
        - Institutions that grant degrees (bachelor's, master's, PhD)
        - Exclude: high schools, elementary schools, research institutes without degree programs, academic departments within universities

        Respond with valid JSON only:
        {{
            "is_university": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                    result_text = response.text.strip()
                    
                    # Clean up the response to extract JSON
                    if result_text.startswith('```json'):
                        result_text = result_text[7:-3]
                    elif result_text.startswith('```'):
                        result_text = result_text[3:-3]
                    
                    result = json.loads(result_text)
                    
                    # Validate response structure
                    if 'is_university' in result and 'confidence' in result:
                        return result
                    else:
                        raise ValueError("Invalid response structure")
                
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Failed to parse Gemini response (attempt {attempt + 1}): {e}")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        logger.error(f"Failed to parse Gemini response after {MAX_RETRIES} attempts: {e}")
                        return {
                            "is_university": False,
                            "confidence": 0.0,
                            "reasoning": "Failed to parse response"
                        }
        
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return {
                "is_university": False,
                "confidence": 0.0,
                "reasoning": f"API error: {str(e)}"
            }
    
    def extract_university_data(self, title: str, extract: str, full_content: str = None) -> Dict:
        """Extract structured university data from Wikipedia content"""
        self._rate_limit()
        
        content = full_content if full_content else extract
        
        prompt = f"""
        Extract detailed information about this university from the Wikipedia content.

        Title: {title}
        Content: {content[:2000]}...

        Extract the following information and respond with valid JSON only:
        {{
            "name": "official university name",
            "city": "city name or null",
            "country": "country name or null",
            "founded": "founding year as integer or null",
            "closed": "closure year as integer or null (if applicable)",
            "type": "public/private/other or null",
            "student_count": "approximate number of students as integer or null",
            "notable_programs": ["list of notable academic programs"],
            "coordinates": {{"lat": latitude, "lng": longitude}} or null
        }}

        Guidelines:
        - Use exact names and spellings from the content
        - For years, extract only the numeric year (e.g., 1887, not "1887 AD")
        - If information is unclear or not mentioned, use null
        - Be conservative with data extraction - only include information you're confident about
        """
        
        try:
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                    result_text = response.text.strip()
                    
                    # Clean up the response to extract JSON
                    if result_text.startswith('```json'):
                        result_text = result_text[7:-3]
                    elif result_text.startswith('```'):
                        result_text = result_text[3:-3]
                    
                    result = json.loads(result_text)
                    
                    # Validate and clean the response
                    cleaned_result = {
                        "name": result.get("name"),
                        "city": result.get("city"),
                        "country": result.get("country"),
                        "founded": result.get("founded") if isinstance(result.get("founded"), int) else None,
                        "closed": result.get("closed") if isinstance(result.get("closed"), int) else None,
                        "type": result.get("type"),
                        "student_count": result.get("student_count") if isinstance(result.get("student_count"), int) else None,
                        "notable_programs": result.get("notable_programs", []) if isinstance(result.get("notable_programs"), list) else [],
                        "coordinates": result.get("coordinates") if isinstance(result.get("coordinates"), dict) else None
                    }
                    
                    return cleaned_result
                
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Failed to parse university data response (attempt {attempt + 1}): {e}")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        logger.error(f"Failed to parse university data after {MAX_RETRIES} attempts: {e}")
                        return {
                            "name": title,
                            "city": None,
                            "country": None,
                            "founded": None,
                            "closed": None,
                            "type": None,
                            "student_count": None,
                            "notable_programs": [],
                            "coordinates": None
                        }
        
        except Exception as e:
            logger.error(f"Error extracting university data: {e}")
            return {
                "name": title,
                "city": None,
                "country": None,
                "founded": None,
                "closed": None,
                "type": None,
                "student_count": None,
                "notable_programs": [],
                "coordinates": None
            }
    
    def batch_university_classification(self, pages: List[Dict]) -> List[Dict]:
        """Classify multiple pages as universities in a single request"""
        if not pages:
            return []
        
        self._rate_limit()
        
        # Prepare batch data
        batch_data = []
        for i, page in enumerate(pages[:10]):  # Limit to 10 pages per batch
            categories_text = ", ".join(page.get('categories', [])) if page.get('categories') else "None"
            batch_data.append({
                "id": i,
                "title": page['title'],
                "extract": page.get('extract', '')[:500],  # Limit extract length
                "categories": categories_text
            })
        
        prompt = f"""
        Classify each of the following Wikipedia pages to determine if they represent universities or higher education institutions.

        Pages to analyze:
        {json.dumps(batch_data, indent=2)}

        For each page, consider:
        - Universities, colleges, and higher education institutions
        - Institutions that grant degrees (bachelor's, master's, PhD)
        - Exclude: high schools, elementary schools, research institutes without degree programs

        Respond with valid JSON only:
        {{
            "results": [
                {{
                    "id": 0,
                    "is_university": true/false,
                    "confidence": 0.0-1.0
                }},
                ...
            ]
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            result_text = response.text.strip()
            
            # Clean up the response
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
            
            result = json.loads(result_text)
            
            # Map results back to original pages
            classifications = []
            for page_result in result.get('results', []):
                page_id = page_result.get('id', 0)
                if page_id < len(pages):
                    classifications.append({
                        'page': pages[page_id],
                        'is_university': page_result.get('is_university', False),
                        'confidence': page_result.get('confidence', 0.0)
                    })
            
            return classifications
        
        except Exception as e:
            logger.error(f"Error in batch classification: {e}")
            # Fallback to individual classification
            results = []
            for page in pages:
                result = self.is_university(
                    page['title'], 
                    page.get('extract', ''), 
                    page.get('categories', [])
                )
                results.append({
                    'page': page,
                    'is_university': result['is_university'],
                    'confidence': result['confidence']
                })
            return results