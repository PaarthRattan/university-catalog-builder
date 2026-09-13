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

                CREATE TABLE IF NOT EXISTS collected_categories (
                    category TEXT PRIMARY KEY,
                    page_count INTEGER,
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            self._migrate(conn)
        logger.info(f"Database initialized at {self.db_path}")

    def _migrate(self, conn):
        """Add columns introduced after the original schema, in place.

        Existing databases keep their data; each column is added only if absent.
        """
        existing = {row[1] for row in conn.execute("PRAGMA table_info(raw_pages)")}
        additions = {
            # How many times a Gemini call for this page has failed. Bounds the
            # filter loop: a page the model never returns a result for is parked
            # after MAX_PAGE_ATTEMPTS instead of being retried forever.
            "attempts": "INTEGER DEFAULT 0",
            # Classification confidence, so the extract stage can apply the same
            # threshold the filter stage used instead of a bare boolean.
            "confidence": "REAL",
            # Extraction checkpoint, separate from the classification checkpoint.
            # Without this, resuming re-extracts every university already done.
            "extracted": "INTEGER DEFAULT 0",
        }
        for col, decl in additions.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE raw_pages ADD COLUMN {col} {decl}")
                logger.info("Migrated raw_pages: added column %s", col)
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_raw_pages_extract_queue
                ON raw_pages(is_university, extracted);
            CREATE INDEX IF NOT EXISTS idx_raw_pages_attempts
                ON raw_pages(processed, attempts);
        """)
    
    def insert_raw_page(self, page_data: Dict) -> bool:
        """Insert raw Wikipedia page data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # ON CONFLICT ... DO UPDATE rather than INSERT OR REPLACE: the
                # latter deletes the row, discarding processed/is_university/
                # extracted, so re-running collection would silently reset every
                # checkpoint and re-pay for all the Gemini work already done.
                conn.execute("""
                    INSERT INTO raw_pages (page_id, title, url, extract, categories)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(page_id) DO UPDATE SET
                        title=excluded.title,
                        url=excluded.url,
                        extract=excluded.extract,
                        categories=excluded.categories
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
    
    def get_unprocessed_pages(self, limit: int = 100, max_attempts: int = 3) -> List[Dict]:
        """Get pages still awaiting classification.

        Pages that have already failed max_attempts times are excluded, which is
        what stops the filter loop from spinning forever on a page the model
        refuses to return a result for.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM raw_pages
                WHERE processed = 0 AND COALESCE(attempts, 0) < ?
                ORDER BY page_id
                LIMIT ?
            """, (max_attempts, limit))
            return [dict(row) for row in cursor.fetchall()]

    def mark_page_processed(self, page_id: int, is_university: bool,
                            confidence: float = None):
        """Mark a page as classified."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE raw_pages
                SET processed = 1, is_university = ?, confidence = ?
                WHERE page_id = ?
            """, (1 if is_university else 0, confidence, page_id))

    def record_failed_attempt(self, page_id: int):
        """Count a failed Gemini attempt without marking the page classified.

        The distinction matters: the old code marked failures as processed with
        is_university=0, permanently mislabelling every page caught by a rate
        limit as 'not a university'.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE raw_pages SET attempts = COALESCE(attempts, 0) + 1
                WHERE page_id = ?
            """, (page_id,))

    def get_pages_for_extraction(self, limit: int = None,
                                 min_confidence: float = 0.0) -> List[Dict]:
        """Universities that still need their details extracted."""
        sql = """
            SELECT page_id, title, url, extract, categories, confidence
            FROM raw_pages
            WHERE is_university = 1
              AND COALESCE(extracted, 0) = 0
              AND COALESCE(confidence, 1.0) >= ?
            ORDER BY page_id
        """
        params = [min_confidence]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def mark_page_extracted(self, page_id: int):
        """Checkpoint a completed extraction so resume never re-pays for it."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE raw_pages SET extracted = 1 WHERE page_id = ?", (page_id,))

    def mark_category_collected(self, category: str, page_count: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO collected_categories (category, page_count)
                VALUES (?, ?)
                ON CONFLICT(category) DO UPDATE SET
                    page_count=excluded.page_count,
                    completed_at=CURRENT_TIMESTAMP
            """, (category, page_count))

    def get_collected_categories(self) -> set:
        with sqlite3.connect(self.db_path) as conn:
            return {r[0] for r in conn.execute("SELECT category FROM collected_categories")}
    
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

            cursor = conn.execute(
                "SELECT COUNT(*) FROM raw_pages WHERE processed = 0 AND COALESCE(attempts,0) < 3")
            stats['pending_classification'] = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM raw_pages WHERE processed = 0 AND COALESCE(attempts,0) >= 3")
            stats['abandoned_pages'] = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM raw_pages WHERE is_university = 1 AND COALESCE(extracted,0) = 0")
            stats['pending_extraction'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM raw_pages WHERE is_university = 1")
            stats['classified_universities'] = cursor.fetchone()[0]
            
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