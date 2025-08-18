#!/usr/bin/env python3
"""
University Catalog Builder
Main application for extracting and cataloging universities from Wikipedia using Gemini AI
"""

import argparse
import logging
import json
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import UniversityExtractionPipeline
from src.deduplication import UniversityDeduplicator
from src.database import UniversityDatabase
from config import DATABASE_PATH, LOG_LEVEL, UNIVERSITY_CATEGORIES

def setup_logging(level: str = LOG_LEVEL):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/university_catalog_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_full_pipeline(args):
    """Run the complete university extraction pipeline"""
    print("🚀 Starting University Catalog Pipeline")
    print("=" * 50)
    
    pipeline = UniversityExtractionPipeline(args.database)
    
    categories = UNIVERSITY_CATEGORIES
    if args.categories:
        categories = args.categories.split(',')
    
    try:
        results = pipeline.run_full_pipeline(categories)
        
        print("\n✅ Pipeline completed successfully!")
        print("=" * 50)
        print(f"📊 Results:")
        print(f"   • Pages collected: {results.get('pages_collected', 0)}")
        print(f"   • Universities filtered: {results.get('universities_filtered', 0)}")
        print(f"   • Universities extracted: {results.get('universities_extracted', 0)}")
        print(f"   • Total processing time: {results.get('total_time_seconds', 0):.2f} seconds")
        
        # Run deduplication if requested
        if not args.skip_dedup:
            print("\n🔄 Running deduplication...")
            deduplicator = UniversityDeduplicator(args.database)
            dedup_results = deduplicator.deduplicate_all()
            print(f"   • Initial count: {dedup_results['initial_count']}")
            print(f"   • Final count: {dedup_results['final_count']}")
            print(f"   • Merged duplicates: {dedup_results['merged_count']}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logging.error(f"Pipeline failed: {e}")
        return None

def run_collection_only(args):
    """Run only the data collection stage"""
    print("📥 Starting data collection only")
    
    pipeline = UniversityExtractionPipeline(args.database)
    categories = UNIVERSITY_CATEGORIES
    if args.categories:
        categories = args.categories.split(',')
    
    pages_collected = pipeline.collect_raw_data(categories)
    print(f"✅ Collected {pages_collected} pages")
    return pages_collected

def run_filtering_only(args):
    """Run only the university filtering stage"""
    print("🔍 Starting university filtering only")
    
    pipeline = UniversityExtractionPipeline(args.database)
    universities_found = pipeline.filter_universities(args.batch_size)
    print(f"✅ Found {universities_found} universities")
    return universities_found

def run_extraction_only(args):
    """Run only the data extraction stage"""
    print("📊 Starting data extraction only")
    
    pipeline = UniversityExtractionPipeline(args.database)
    universities_extracted = pipeline.extract_university_details()
    print(f"✅ Extracted {universities_extracted} university details")
    return universities_extracted

def run_deduplication(args):
    """Run deduplication process"""
    print("🔄 Starting deduplication")
    
    deduplicator = UniversityDeduplicator(args.database)
    
    if args.report_only:
        # Generate duplicate report
        report = deduplicator.get_duplicate_report()
        output_file = f"duplicate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Duplicate report saved to {output_file}")
        print(f"   Found {len(report)} potential duplicate groups")
        return report
    else:
        # Run actual deduplication
        results = deduplicator.deduplicate_all()
        print(f"✅ Deduplication completed:")
        print(f"   • Initial count: {results['initial_count']}")
        print(f"   • Final count: {results['final_count']}")
        print(f"   • Merged duplicates: {results['merged_count']}")
        return results

def show_statistics(args):
    """Show current database statistics"""
    print("📈 Database Statistics")
    print("=" * 30)
    
    db = UniversityDatabase(args.database)
    stats = db.get_statistics()
    
    print(f"Total pages: {stats.get('total_pages', 0)}")
    print(f"Processed pages: {stats.get('processed_pages', 0)}")
    print(f"Universities: {stats.get('universities', 0)}")
    
    if stats.get('top_countries'):
        print(f"\nTop countries by university count:")
        for country, count in stats['top_countries'][:10]:
            print(f"  {country}: {count}")
    
    return stats

def export_data(args):
    """Export university data to various formats"""
    print(f"📤 Exporting data to {args.output}")
    
    import sqlite3
    import pandas as pd
    
    try:
        # Connect to database and export
        with sqlite3.connect(args.database) as conn:
            df = pd.read_sql_query("""
                SELECT name, city, country, founded_year, closed_year, 
                       university_type, wikipedia_url, created_at
                FROM universities 
                ORDER BY country, city, name
            """, conn)
        
        # Export based on file extension
        if args.output.endswith('.csv'):
            df.to_csv(args.output, index=False)
        elif args.output.endswith('.json'):
            df.to_json(args.output, orient='records', indent=2)
        elif args.output.endswith('.xlsx'):
            df.to_excel(args.output, index=False)
        else:
            # Default to CSV
            df.to_csv(args.output + '.csv', index=False)
        
        print(f"✅ Exported {len(df)} universities to {args.output}")
        return len(df)
    
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 0

def resume_pipeline(args):
    """Resume pipeline from last checkpoint"""
    print("🔄 Resuming pipeline from last checkpoint")
    
    pipeline = UniversityExtractionPipeline(args.database)
    results = pipeline.resume_pipeline()
    
    print("✅ Pipeline resumed successfully")
    print(f"📊 Current status:")
    for key, value in results.items():
        print(f"   • {key}: {value}")
    
    return results

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="University Catalog Builder")
    parser.add_argument('--database', default=DATABASE_PATH, help='Database file path')
    parser.add_argument('--log-level', default=LOG_LEVEL, help='Logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full pipeline
    full_parser = subparsers.add_parser('run', help='Run complete pipeline')
    full_parser.add_argument('--categories', help='Comma-separated list of categories')
    full_parser.add_argument('--skip-dedup', action='store_true', help='Skip deduplication')
    
    # Individual stages
    collect_parser = subparsers.add_parser('collect', help='Run data collection only')
    collect_parser.add_argument('--categories', help='Comma-separated list of categories')
    
    filter_parser = subparsers.add_parser('filter', help='Run university filtering only')
    filter_parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    
    extract_parser = subparsers.add_parser('extract', help='Run data extraction only')
    
    # Deduplication
    dedup_parser = subparsers.add_parser('dedupe', help='Run deduplication')
    dedup_parser.add_argument('--report-only', action='store_true', help='Generate report only')
    
    # Utilities
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    export_parser = subparsers.add_parser('export', help='Export data')
    export_parser.add_argument('output', help='Output file path')
    
    resume_parser = subparsers.add_parser('resume', help='Resume pipeline')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.database), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Route to appropriate function
    if args.command == 'run':
        return run_full_pipeline(args)
    elif args.command == 'collect':
        return run_collection_only(args)
    elif args.command == 'filter':
        return run_filtering_only(args)
    elif args.command == 'extract':
        return run_extraction_only(args)
    elif args.command == 'dedupe':
        return run_deduplication(args)
    elif args.command == 'stats':
        return show_statistics(args)
    elif args.command == 'export':
        return export_data(args)
    elif args.command == 'resume':
        return resume_pipeline(args)
    else:
        parser.print_help()
        return None

if __name__ == '__main__':
    main()