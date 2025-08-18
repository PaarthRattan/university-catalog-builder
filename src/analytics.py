import sqlite3
import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class UniversityAnalytics:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_universities_per_capita_by_year(self, year: int, population_data: Dict[str, int] = None) -> pd.DataFrame:
        """
        Calculate universities per capita by country for a given year
        
        Args:
            year: Target year for analysis
            population_data: Dictionary mapping country names to population counts
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get universities that existed in the target year
            query = """
                SELECT country, COUNT(*) as university_count
                FROM universities 
                WHERE country IS NOT NULL 
                AND (founded_year IS NULL OR founded_year <= ?)
                AND (closed_year IS NULL OR closed_year > ?)
                GROUP BY country
                ORDER BY university_count DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[year, year])
        
        if population_data:
            # Add population data and calculate per capita
            df['population'] = df['country'].map(population_data)
            df['universities_per_million'] = (df['university_count'] / df['population']) * 1_000_000
            df = df.dropna(subset=['population']).sort_values('universities_per_million', ascending=False)
        
        return df
    
    def get_university_founding_timeline(self) -> pd.DataFrame:
        """Get timeline of university founding by decade"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    (founded_year / 10) * 10 as decade,
                    COUNT(*) as universities_founded,
                    country
                FROM universities 
                WHERE founded_year IS NOT NULL
                GROUP BY decade, country
                ORDER BY decade, universities_founded DESC
            """
            
            return pd.read_sql_query(query, conn)
    
    def get_country_statistics(self) -> pd.DataFrame:
        """Get comprehensive statistics by country"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    country,
                    COUNT(*) as total_universities,
                    MIN(founded_year) as oldest_university_year,
                    MAX(founded_year) as newest_university_year,
                    AVG(founded_year) as avg_founding_year,
                    COUNT(CASE WHEN university_type = 'public' THEN 1 END) as public_count,
                    COUNT(CASE WHEN university_type = 'private' THEN 1 END) as private_count,
                    COUNT(CASE WHEN closed_year IS NOT NULL THEN 1 END) as closed_count
                FROM universities 
                WHERE country IS NOT NULL
                GROUP BY country
                HAVING total_universities >= 5
                ORDER BY total_universities DESC
            """
            
            return pd.read_sql_query(query, conn)
    
    def get_universities_by_type(self) -> pd.DataFrame:
        """Analyze universities by type (public/private)"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    university_type,
                    country,
                    COUNT(*) as count,
                    AVG(founded_year) as avg_founding_year
                FROM universities 
                WHERE university_type IS NOT NULL AND country IS NOT NULL
                GROUP BY university_type, country
                ORDER BY country, count DESC
            """
            
            return pd.read_sql_query(query, conn)
    
    def get_temporal_analysis(self, start_year: int = 1000, end_year: int = None) -> Dict:
        """
        Comprehensive temporal analysis of universities
        
        Args:
            start_year: Starting year for analysis
            end_year: Ending year (defaults to current year)
        """
        if not end_year:
            end_year = datetime.now().year
        
        with sqlite3.connect(self.db_path) as conn:
            # Universities active by year
            active_by_year = {}
            
            for year in range(start_year, end_year + 1, 10):  # Every 10 years
                query = """
                    SELECT country, COUNT(*) as count
                    FROM universities 
                    WHERE country IS NOT NULL
                    AND (founded_year IS NULL OR founded_year <= ?)
                    AND (closed_year IS NULL OR closed_year > ?)
                    GROUP BY country
                """
                
                df = pd.read_sql_query(query, conn, params=[year, year])
                active_by_year[year] = df.to_dict('records')
            
            # Founding trends by century
            century_query = """
                SELECT 
                    CASE 
                        WHEN founded_year < 1500 THEN 'Before 1500'
                        WHEN founded_year < 1600 THEN '1500-1599'
                        WHEN founded_year < 1700 THEN '1600-1699'
                        WHEN founded_year < 1800 THEN '1700-1799'
                        WHEN founded_year < 1900 THEN '1800-1899'
                        WHEN founded_year < 2000 THEN '1900-1999'
                        ELSE '2000+'
                    END as period,
                    COUNT(*) as universities_founded
                FROM universities 
                WHERE founded_year IS NOT NULL
                GROUP BY period
                ORDER BY MIN(founded_year)
            """
            
            founding_trends = pd.read_sql_query(century_query, conn)
            
            return {
                'active_by_year': active_by_year,
                'founding_trends': founding_trends.to_dict('records'),
                'analysis_period': f"{start_year}-{end_year}"
            }
    
    def get_geographic_distribution(self) -> Dict:
        """Analyze geographic distribution of universities"""
        with sqlite3.connect(self.db_path) as conn:
            # By country
            country_query = """
                SELECT country, COUNT(*) as count
                FROM universities 
                WHERE country IS NOT NULL
                GROUP BY country
                ORDER BY count DESC
            """
            
            by_country = pd.read_sql_query(country_query, conn)
            
            # By city (top cities)
            city_query = """
                SELECT city, country, COUNT(*) as count
                FROM universities 
                WHERE city IS NOT NULL AND country IS NOT NULL
                GROUP BY city, country
                HAVING count >= 3
                ORDER BY count DESC
                LIMIT 50
            """
            
            by_city = pd.read_sql_query(city_query, conn)
            
            return {
                'by_country': by_country.to_dict('records'),
                'by_city': by_city.to_dict('records'),
                'total_countries': len(by_country),
                'total_cities': len(by_city)
            }
    
    def generate_comprehensive_report(self, output_file: str = None) -> Dict:
        """Generate a comprehensive analytical report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'geographic_distribution': {},
            'temporal_analysis': {},
            'university_types': {},
            'top_countries': {},
            'recommendations': []
        }
        
        try:
            # Basic summary
            with sqlite3.connect(self.db_path) as conn:
                summary_query = """
                    SELECT 
                        COUNT(*) as total_universities,
                        COUNT(DISTINCT country) as countries_covered,
                        COUNT(DISTINCT city) as cities_covered,
                        MIN(founded_year) as oldest_university,
                        MAX(founded_year) as newest_university,
                        COUNT(CASE WHEN closed_year IS NOT NULL THEN 1 END) as closed_universities
                    FROM universities
                """
                
                summary_df = pd.read_sql_query(summary_query, conn)
                report['summary'] = summary_df.iloc[0].to_dict()
            
            # Geographic analysis
            report['geographic_distribution'] = self.get_geographic_distribution()
            
            # Temporal analysis
            report['temporal_analysis'] = self.get_temporal_analysis()
            
            # University types
            types_df = self.get_universities_by_type()
            report['university_types'] = types_df.to_dict('records')
            
            # Top countries analysis
            countries_df = self.get_country_statistics()
            report['top_countries'] = countries_df.head(20).to_dict('records')
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
            # Save report if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Comprehensive report saved to {output_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate data quality and analysis recommendations"""
        recommendations = []
        
        summary = report.get('summary', {})
        total_unis = summary.get('total_universities', 0)
        
        # Data completeness recommendations
        if total_unis > 0:
            # Check for missing data
            with sqlite3.connect(self.db_path) as conn:
                missing_data_query = """
                    SELECT 
                        COUNT(CASE WHEN country IS NULL THEN 1 END) * 100.0 / COUNT(*) as missing_country_pct,
                        COUNT(CASE WHEN city IS NULL THEN 1 END) * 100.0 / COUNT(*) as missing_city_pct,
                        COUNT(CASE WHEN founded_year IS NULL THEN 1 END) * 100.0 / COUNT(*) as missing_founded_pct,
                        COUNT(CASE WHEN university_type IS NULL THEN 1 END) * 100.0 / COUNT(*) as missing_type_pct
                    FROM universities
                """
                
                missing_df = pd.read_sql_query(missing_data_query, conn)
                missing = missing_df.iloc[0]
                
                if missing['missing_country_pct'] > 10:
                    recommendations.append(f"High percentage ({missing['missing_country_pct']:.1f}%) of universities missing country data")
                
                if missing['missing_founded_pct'] > 20:
                    recommendations.append(f"Consider improving founding year extraction ({missing['missing_founded_pct']:.1f}% missing)")
                
                if missing['missing_type_pct'] > 50:
                    recommendations.append("University type classification needs improvement - consider additional LLM prompts")
        
        # Coverage recommendations
        geographic = report.get('geographic_distribution', {})
        if geographic.get('total_countries', 0) < 50:
            recommendations.append("Consider expanding Wikipedia category search to cover more countries")
        
        # Temporal analysis recommendations
        temporal = report.get('temporal_analysis', {})
        founding_trends = temporal.get('founding_trends', [])
        if founding_trends:
            recent_founding = next((item for item in founding_trends if item['period'] == '2000+'), {})
            if recent_founding.get('universities_founded', 0) < 100:
                recommendations.append("Recent university data may be incomplete - consider additional search terms")
        
        return recommendations
    
    def export_analysis_data(self, output_dir: str = "analysis_output"):
        """Export all analysis data to separate files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Country statistics
            country_stats = self.get_country_statistics()
            country_stats.to_csv(f"{output_dir}/country_statistics.csv", index=False)
            
            # Founding timeline
            timeline = self.get_university_founding_timeline()
            timeline.to_csv(f"{output_dir}/founding_timeline.csv", index=False)
            
            # University types
            types = self.get_universities_by_type()
            types.to_csv(f"{output_dir}/university_types.csv", index=False)
            
            # Geographic distribution
            geo_dist = self.get_geographic_distribution()
            with open(f"{output_dir}/geographic_distribution.json", 'w') as f:
                json.dump(geo_dist, f, indent=2)
            
            # Temporal analysis
            temporal = self.get_temporal_analysis()
            with open(f"{output_dir}/temporal_analysis.json", 'w') as f:
                json.dump(temporal, f, indent=2, default=str)
            
            # Comprehensive report
            report = self.generate_comprehensive_report()
            with open(f"{output_dir}/comprehensive_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Analysis data exported to {output_dir}/")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting analysis data: {e}")
            return False