import sqlite3
from typing import List, Dict, Set, Tuple
import re
from fuzzywuzzy import fuzz, process
import logging

logger = logging.getLogger(__name__)

class UniversityDeduplicator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.similarity_threshold = 85  # Minimum similarity score for duplicates
    
    def normalize_name(self, name: str) -> str:
        """Normalize university name for comparison"""
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove common prefixes/suffixes
        patterns_to_remove = [
            r'\buniversity of\b',
            r'\buniversity\b$',
            r'\bcollege of\b',
            r'\bcollege\b$',
            r'\binstitute of\b',
            r'\binstitute\b$',
            r'\bthe\b',
            r'\b(state|national|public|private)\b',
            r'\b(technical|medical|business|art)\b',
            r'\b(school|academy)\b',
            r'\bcampus\b',
            r'\bmain\b',
            r'\bcentral\b'
        ]
        
        for pattern in patterns_to_remove:
            normalized = re.sub(pattern, '', normalized)
        
        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def extract_location_key(self, city: str, country: str) -> str:
        """Create a location key for grouping"""
        city_clean = self.normalize_name(city) if city else ""
        country_clean = self.normalize_name(country) if country else ""
        return f"{country_clean}_{city_clean}"
    
    def find_potential_duplicates(self) -> List[List[Dict]]:
        """Find groups of potentially duplicate universities"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, name, city, country, founded_year, wikipedia_url
                FROM universities
                ORDER BY country, city, name
            """)
            universities = [dict(row) for row in cursor.fetchall()]
        
        # Group by location first to reduce comparison space
        location_groups = {}
        for uni in universities:
            location_key = self.extract_location_key(uni['city'], uni['country'])
            if location_key not in location_groups:
                location_groups[location_key] = []
            location_groups[location_key].append(uni)
        
        duplicate_groups = []
        
        for location, unis in location_groups.items():
            if len(unis) < 2:
                continue
            
            # Find duplicates within each location group
            location_duplicates = self._find_duplicates_in_group(unis)
            duplicate_groups.extend(location_duplicates)
        
        logger.info(f"Found {len(duplicate_groups)} potential duplicate groups")
        return duplicate_groups
    
    def _find_duplicates_in_group(self, universities: List[Dict]) -> List[List[Dict]]:
        """Find duplicates within a group of universities"""
        duplicate_groups = []
        processed_ids = set()
        
        for i, uni1 in enumerate(universities):
            if uni1['id'] in processed_ids:
                continue
            
            current_group = [uni1]
            processed_ids.add(uni1['id'])
            
            for j, uni2 in enumerate(universities[i+1:], i+1):
                if uni2['id'] in processed_ids:
                    continue
                
                if self._are_duplicates(uni1, uni2):
                    current_group.append(uni2)
                    processed_ids.add(uni2['id'])
            
            # Only add groups with actual duplicates
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
        
        return duplicate_groups
    
    def _are_duplicates(self, uni1: Dict, uni2: Dict) -> bool:
        """Determine if two universities are duplicates"""
        # Normalize names
        name1 = self.normalize_name(uni1['name'])
        name2 = self.normalize_name(uni2['name'])
        
        # Calculate name similarity
        name_similarity = fuzz.ratio(name1, name2)
        
        # Check if names are very similar
        if name_similarity >= self.similarity_threshold:
            return True
        
        # Check for exact matches in different formats
        if self._check_name_variants(uni1['name'], uni2['name']):
            return True
        
        # Check if one name is contained in the other (with some conditions)
        if self._check_name_containment(name1, name2):
            return True
        
        # Check founding year if available
        if (uni1['founded_year'] and uni2['founded_year'] and 
            uni1['founded_year'] == uni2['founded_year'] and
            name_similarity >= 70):
            return True
        
        return False
    
    def _check_name_variants(self, name1: str, name2: str) -> bool:
        """Check for common name variants"""
        # Convert to sets of words for comparison
        words1 = set(self.normalize_name(name1).split())
        words2 = set(self.normalize_name(name2).split())
        
        # Remove very common words
        common_words = {'of', 'the', 'and', 'at', 'in', 'for'}
        words1 -= common_words
        words2 -= common_words
        
        if not words1 or not words2:
            return False
        
        # Check if one is a subset of the other
        if words1.issubset(words2) or words2.issubset(words1):
            return True
        
        # Check intersection ratio
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union > 0 and intersection / union >= 0.8:
            return True
        
        return False
    
    def _check_name_containment(self, name1: str, name2: str) -> bool:
        """Check if one name is contained in another"""
        if len(name1) < 3 or len(name2) < 3:
            return False
        
        longer, shorter = (name1, name2) if len(name1) > len(name2) else (name2, name1)
        
        # Only consider containment if the shorter name is substantial
        if len(shorter) >= 5 and shorter in longer:
            # Additional check: make sure it's not just a common word
            if len(shorter.split()) >= 2:
                return True
        
        return False
    
    def merge_duplicates(self, duplicate_groups: List[List[Dict]]) -> int:
        """Merge duplicate university entries"""
        merged_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for group in duplicate_groups:
                if len(group) < 2:
                    continue
                
                # Choose the best entry as the primary
                primary = self._choose_primary_entry(group)
                duplicates = [uni for uni in group if uni['id'] != primary['id']]
                
                # Merge data into primary entry
                merged_data = self._merge_university_data(primary, duplicates)
                
                # Update primary entry
                self._update_university(conn, primary['id'], merged_data)
                
                # Delete duplicate entries
                for duplicate in duplicates:
                    conn.execute("DELETE FROM universities WHERE id = ?", (duplicate['id'],))
                    logger.debug(f"Deleted duplicate: {duplicate['name']}")
                
                merged_count += len(duplicates)
                logger.info(f"Merged {len(duplicates)} duplicates into: {primary['name']}")
        
        logger.info(f"Merged {merged_count} duplicate entries")
        return merged_count
    
    def _choose_primary_entry(self, group: List[Dict]) -> Dict:
        """Choose the best entry from a group of duplicates"""
        # Scoring criteria (higher is better)
        def score_entry(uni):
            score = 0
            
            # Prefer entries with more complete data
            if uni['city']:
                score += 10
            if uni['country']:
                score += 10
            if uni['founded_year']:
                score += 5
            
            # Prefer longer, more detailed names
            score += len(uni['name']) * 0.1
            
            # Prefer entries with "University" in the name
            if 'university' in uni['name'].lower():
                score += 5
            
            return score
        
        return max(group, key=score_entry)
    
    def _merge_university_data(self, primary: Dict, duplicates: List[Dict]) -> Dict:
        """Merge data from duplicates into primary entry"""
        merged = dict(primary)
        
        for duplicate in duplicates:
            # Fill in missing data from duplicates
            if not merged['city'] and duplicate['city']:
                merged['city'] = duplicate['city']
            if not merged['country'] and duplicate['country']:
                merged['country'] = duplicate['country']
            if not merged['founded_year'] and duplicate['founded_year']:
                merged['founded_year'] = duplicate['founded_year']
            
            # Choose the longer, more descriptive name
            if len(duplicate['name']) > len(merged['name']):
                merged['name'] = duplicate['name']
        
        return merged
    
    def _update_university(self, conn, university_id: int, data: Dict):
        """Update university entry in database"""
        conn.execute("""
            UPDATE universities 
            SET name = ?, city = ?, country = ?, founded_year = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (data['name'], data['city'], data['country'], data['founded_year'], university_id))
    
    def deduplicate_all(self) -> Dict:
        """Run complete deduplication process"""
        logger.info("Starting university deduplication")
        
        # Get initial count
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM universities")
            initial_count = cursor.fetchone()[0]
        
        # Find and merge duplicates
        duplicate_groups = self.find_potential_duplicates()
        merged_count = self.merge_duplicates(duplicate_groups)
        
        # Get final count
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM universities")
            final_count = cursor.fetchone()[0]
        
        results = {
            'initial_count': initial_count,
            'final_count': final_count,
            'merged_count': merged_count,
            'duplicate_groups_found': len(duplicate_groups)
        }
        
        logger.info(f"Deduplication completed: {initial_count} -> {final_count} universities ({merged_count} merged)")
        return results
    
    def get_duplicate_report(self) -> List[Dict]:
        """Generate a report of potential duplicates for manual review"""
        duplicate_groups = self.find_potential_duplicates()
        
        report = []
        for i, group in enumerate(duplicate_groups):
            group_report = {
                'group_id': i + 1,
                'universities': [],
                'similarity_scores': []
            }
            
            for uni in group:
                group_report['universities'].append({
                    'id': uni['id'],
                    'name': uni['name'],
                    'city': uni['city'],
                    'country': uni['country'],
                    'founded_year': uni['founded_year'],
                    'url': uni['wikipedia_url']
                })
            
            # Calculate pairwise similarities
            for i, uni1 in enumerate(group):
                for uni2 in group[i+1:]:
                    similarity = fuzz.ratio(
                        self.normalize_name(uni1['name']),
                        self.normalize_name(uni2['name'])
                    )
                    group_report['similarity_scores'].append({
                        'pair': [uni1['name'], uni2['name']],
                        'similarity': similarity
                    })
            
            report.append(group_report)
        
        return report