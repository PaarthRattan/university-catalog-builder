import sqlite3
import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class UniversityDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS universities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    wikipedia_url TEXT UNIQUE NOT NULL,
                    city TEXT,
                    country TEXT,
                    founded_year INTEGER,
                    closed_year INTEGER,
                    university_type TEXT,
                    is_verified BOOLEAN DEFAULT 0,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS raw_pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_id INTEGER UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    extract TEXT,
                    categories TEXT,
                    processed BOOLEAN DEFAULT 0,
                    is_university BOOLEAN DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    page_count INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_universities_country ON universities(country);
                CREATE INDEX IF NOT EXISTS idx_universities_founded ON universities(founded_year);
                CREATE INDEX IF NOT EXISTS idx_raw_pages_processed ON raw_pages(processed);
            """)
        logger.info(f"Database initialized at {self.db_path}")
    
    def insert_raw_page(self, page_data: Dict) -> bool:
        """Insert raw Wikipedia page data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO raw_pages 
                    (page_id, title, url, extract, categories)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    page_data['page_id'],
                    page_data['title'],
                    page_data['url'],
                    page_data.get('extract', ''),
                    ','.join(page_data.get('categories', []))
                ))
            return True
        except Exception as e:
            logger.error(f"Error inserting raw page: {e}")
            return False
    
    def insert_university(self, university_data: Dict) -> bool:
        """Insert verified university data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO universities 
                    (name, wikipedia_url, city, country, founded_year, closed_year, 
                     university_type, is_verified, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    university_data['name'],
                    university_data['wikipedia_url'],
                    university_data.get('city'),
                    university_data.get('country'),
                    university_data.get('founded_year'),
                    university_data.get('closed_year'),
                    university_data.get('university_type'),
                    True,
                    university_data.get('confidence_score', 1.0)
                ))
            return True
        except Exception as e:
            logger.error(f"Error inserting university: {e}")
            return False
    
    def get_unprocessed_pages(self, limit: int = 100) -> List[Dict]:
        """Get unprocessed Wikipedia pages"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM raw_pages 
                WHERE processed = 0 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def mark_page_processed(self, page_id: int, is_university: bool):
        """Mark a page as processed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE raw_pages 
                SET processed = 1, is_university = ? 
                WHERE page_id = ?
            """, (is_university, page_id))
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Raw pages count
            cursor = conn.execute("SELECT COUNT(*) FROM raw_pages")
            stats['total_pages'] = cursor.fetchone()[0]
            
            # Processed pages count
            cursor = conn.execute("SELECT COUNT(*) FROM raw_pages WHERE processed = 1")
            stats['processed_pages'] = cursor.fetchone()[0]
            
            # Universities count
            cursor = conn.execute("SELECT COUNT(*) FROM universities")
            stats['universities'] = cursor.fetchone()[0]
            
            # Universities by country
            cursor = conn.execute("""
                SELECT country, COUNT(*) as count 
                FROM universities 
                WHERE country IS NOT NULL 
                GROUP BY country 
                ORDER BY count DESC 
                LIMIT 10
            """)
            stats['top_countries'] = cursor.fetchall()
            
        return stats
    
    def log_processing_stage(self, stage: str, status: str, message: str = None, page_count: int = None):
        """Log processing stage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO processing_log (stage, status, message, page_count)
                VALUES (?, ?, ?, ?)
            """, (stage, status, message, page_count))