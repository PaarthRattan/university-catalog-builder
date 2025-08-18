import asyncio
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
import time

from .wikipedia_client import WikipediaClient
from .gemini_client import GeminiClient
from .database import UniversityDatabase
from config import UNIVERSITY_CATEGORIES, BATCH_SIZE, DATABASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversityExtractionPipeline:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.wikipedia = WikipediaClient()
        self.gemini = GeminiClient()
        self.db = UniversityDatabase(db_path)
        
    def collect_raw_data(self, categories: List[str] = None) -> int:
        """Stage 1: Collect raw Wikipedia pages from university categories"""
        if not categories:
            categories = UNIVERSITY_CATEGORIES
        
        logger.info(f"Starting data collection from {len(categories)} categories")
        self.db.log_processing_stage("data_collection", "started", f"Processing {len(categories)} categories")
        
        total_pages = 0
        all_pages = set()  # Use set to avoid duplicates
        
        for category in tqdm(categories, desc="Processing categories"):
            try:
                # Get main category pages
                pages = self.wikipedia.get_category_members(category, limit=1000)
                
                # Get subcategories and their pages
                subcategories = self.wikipedia.get_subcategories(category)
                for subcat in subcategories[:5]:  # Limit subcategories to avoid explosion
                    sub_pages = self.wikipedia.get_category_members(subcat, limit=500)
                    pages.extend(sub_pages)
                
                # Add to set (automatically deduplicates by page_id)
                for page in pages:
                    all_pages.add((page['page_id'], page['title'], page['url']))
                
                logger.info(f"Category {category}: {len(pages)} pages found")
                
            except Exception as e:
                logger.error(f"Error processing category {category}: {e}")
                continue
        
        # Convert back to list and get page extracts
        unique_pages = [{'page_id': pid, 'title': title, 'url': url} 
                       for pid, title, url in all_pages]
        
        logger.info(f"Found {len(unique_pages)} unique pages, getting extracts...")
        
        # Get page extracts in batches
        page_ids = [page['page_id'] for page in unique_pages]
        extracts = self.wikipedia.get_page_extracts(page_ids)
        
        # Store raw pages in database
        for page in tqdm(unique_pages, desc="Storing raw pages"):
            page_data = {
                'page_id': page['page_id'],
                'title': page['title'],
                'url': page['url'],
                'extract': extracts.get(page['page_id'], {}).get('extract', ''),
                'categories': extracts.get(page['page_id'], {}).get('categories', [])
            }
            
            if self.db.insert_raw_page(page_data):
                total_pages += 1
        
        self.db.log_processing_stage("data_collection", "completed", f"Collected {total_pages} pages")
        logger.info(f"Data collection completed: {total_pages} pages stored")
        return total_pages
    
    def filter_universities(self, batch_size: int = 10) -> int:
        """Stage 2: Use Gemini to filter pages and identify universities"""
        logger.info("Starting university filtering with Gemini")
        self.db.log_processing_stage("university_filtering", "started")
        
        universities_found = 0
        processed_count = 0
        
        while True:
            # Get batch of unprocessed pages
            unprocessed = self.db.get_unprocessed_pages(batch_size)
            if not unprocessed:
                break
            
            try:
                # Use batch classification for efficiency
                classifications = self.gemini.batch_university_classification(unprocessed)
                
                for classification in classifications:
                    page = classification['page']
                    is_university = classification['is_university']
                    confidence = classification['confidence']
                    
                    # Mark page as processed
                    self.db.mark_page_processed(page['page_id'], is_university)
                    processed_count += 1
                    
                    if is_university and confidence > 0.5:
                        universities_found += 1
                        logger.debug(f"University found: {page['title']} (confidence: {confidence:.2f})")
                
                logger.info(f"Processed batch: {len(classifications)} pages, {sum(c['is_university'] for c in classifications)} universities")
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Mark pages as processed even if failed to avoid infinite loop
                for page in unprocessed:
                    self.db.mark_page_processed(page['page_id'], False)
                processed_count += len(unprocessed)
        
        self.db.log_processing_stage("university_filtering", "completed", f"Found {universities_found} universities from {processed_count} pages")
        logger.info(f"University filtering completed: {universities_found} universities found from {processed_count} pages")
        return universities_found
    
    def extract_university_details(self) -> int:
        """Stage 3: Extract detailed information for identified universities"""
        logger.info("Starting detailed university data extraction")
        self.db.log_processing_stage("data_extraction", "started")
        
        # Get pages marked as universities
        import sqlite3
        with sqlite3.connect(self.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT page_id, title, url, extract, categories 
                FROM raw_pages 
                WHERE is_university = 1
            """)
            university_pages = cursor.fetchall()
        
        extracted_count = 0
        
        for page in tqdm(university_pages, desc="Extracting university details"):
            try:
                # Extract detailed university data
                university_data = self.gemini.extract_university_data(
                    page['title'],
                    page['extract']
                )
                
                # Prepare data for database
                db_data = {
                    'name': university_data['name'] or page['title'],
                    'wikipedia_url': page['url'],
                    'city': university_data['city'],
                    'country': university_data['country'],
                    'founded_year': university_data['founded'],
                    'closed_year': university_data['closed'],
                    'university_type': university_data['type'],
                    'confidence_score': 1.0  # High confidence since already filtered
                }
                
                if self.db.insert_university(db_data):
                    extracted_count += 1
                    logger.debug(f"Extracted: {university_data['name']} ({university_data['country']})")
                
            except Exception as e:
                logger.error(f"Error extracting data for {page['title']}: {e}")
                continue
        
        self.db.log_processing_stage("data_extraction", "completed", f"Extracted details for {extracted_count} universities")
        logger.info(f"Data extraction completed: {extracted_count} universities processed")
        return extracted_count
    
    def run_full_pipeline(self, categories: List[str] = None) -> Dict:
        """Run the complete pipeline"""
        start_time = time.time()
        logger.info("Starting full university extraction pipeline")
        
        results = {}
        
        try:
            # Stage 1: Collect raw data
            results['pages_collected'] = self.collect_raw_data(categories)
            
            # Stage 2: Filter universities
            results['universities_filtered'] = self.filter_universities()
            
            # Stage 3: Extract details
            results['universities_extracted'] = self.extract_university_details()
            
            # Get final statistics
            stats = self.db.get_statistics()
            results.update(stats)
            
            elapsed_time = time.time() - start_time
            results['total_time_seconds'] = elapsed_time
            
            logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
            logger.info(f"Final results: {results}")
            
            self.db.log_processing_stage("full_pipeline", "completed", f"Pipeline completed successfully", results['universities_extracted'])
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.db.log_processing_stage("full_pipeline", "failed", str(e))
            raise
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get current pipeline statistics"""
        return self.db.get_statistics()
    
    def resume_pipeline(self) -> Dict:
        """Resume pipeline from where it left off"""
        logger.info("Resuming pipeline from last checkpoint")
        
        stats = self.db.get_statistics()
        
        # Check what stage we're at
        if stats['processed_pages'] < stats['total_pages']:
            logger.info("Resuming from university filtering stage")
            universities_filtered = self.filter_universities()
            universities_extracted = self.extract_university_details()
        elif stats['universities'] == 0:
            logger.info("Resuming from data extraction stage")
            universities_extracted = self.extract_university_details()
        else:
            logger.info("Pipeline appears to be complete")
            return self.get_statistics()
        
        return self.get_statistics()

def _get_connection(self):
    """Helper method for database connection"""
    import sqlite3
    return sqlite3.connect(self.db_path)