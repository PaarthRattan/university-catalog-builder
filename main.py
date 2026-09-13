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

from src.pipeline import UniversityExtractionPipeline, PipelineHalted
from src.deduplication import UniversityDeduplicator
from src.database import UniversityDatabase
from src.metrics import MetricsCollector
from config import (DATABASE_PATH, LOG_LEVEL, UNIVERSITY_CATEGORIES,
                    BENCHMARK_CATEGORIES, METRICS_DIR, GEMINI_MODEL,
                    GEMINI_RPM, GEMINI_TPM, GEMINI_RPD,
                    GEMINI_CLASSIFY_BATCH_SIZE, GEMINI_COMBINED_BATCH_SIZE,
                    GEMINI_COMBINED_FILTER_EXTRACT, GEMINI_MAX_CONCURRENCY,
                    GEMINI_PRICE_INPUT_PER_MTOK, GEMINI_PRICE_OUTPUT_PER_MTOK)

# Exit code used when the run stops cleanly at the daily quota boundary, so a
# wrapper script can tell "out of quota, resume later" from "crashed".
EXIT_QUOTA_EXHAUSTED = 2

def setup_logging(level: str = LOG_LEVEL):
    """Setup logging configuration"""
    # Create the log directory before the FileHandler opens a file inside it --
    # otherwise every invocation dies with FileNotFoundError before doing work.
    os.makedirs('logs', exist_ok=True)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/university_catalog_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def _combined_flag(args):
    """Resolve the combined filter+extract A/B setting for this invocation."""
    if getattr(args, 'combined', False):
        return True
    if getattr(args, 'two_pass', False):
        return False
    return GEMINI_COMBINED_FILTER_EXTRACT


def _make_metrics(args, label: str, combined: bool):
    if getattr(args, 'no_metrics', False):
        return None
    return MetricsCollector(
        run_label=getattr(args, 'run_label', None) or label,
        metrics_dir=getattr(args, 'metrics_dir', None) or METRICS_DIR,
        price_input_per_mtok=GEMINI_PRICE_INPUT_PER_MTOK,
        price_output_per_mtok=GEMINI_PRICE_OUTPUT_PER_MTOK,
        config_snapshot={
            'model': GEMINI_MODEL,
            'rpm_limit': GEMINI_RPM,
            'tpm_limit': GEMINI_TPM,
            'rpd_limit': GEMINI_RPD,
            'classify_batch_size': GEMINI_CLASSIFY_BATCH_SIZE,
            'combined_batch_size': GEMINI_COMBINED_BATCH_SIZE,
            'combined_filter_extract': combined,
            'max_concurrency': min(GEMINI_MAX_CONCURRENCY, GEMINI_RPM),
            'categories': getattr(args, 'categories', None),
        })


def _finish(metrics, pipeline=None, extra=None):
    """Write the metrics file and print the summary table."""
    if not metrics:
        return None
    payload = dict(extra or {})
    if pipeline is not None:
        try:
            payload['rate_limiter'] = pipeline.gemini.limiter_snapshot()
        except Exception:
            pass
    path = metrics.finalize(payload)
    metrics.print_summary()
    print(f"\nMetrics written to: {path}")
    return path


def run_full_pipeline(args):
    """Run the complete university extraction pipeline"""
    print("🚀 Starting University Catalog Pipeline")
    print("=" * 50)
    
    combined = _combined_flag(args)
    metrics = _make_metrics(args, 'full_run', combined)
    pipeline = UniversityExtractionPipeline(args.database, metrics=metrics,
                                            combined=combined)

    categories = UNIVERSITY_CATEGORIES
    if args.benchmark:
        categories = BENCHMARK_CATEGORIES
    elif args.categories:
        categories = args.categories.split(',')

    print(f"   Path: {'combined filter+extract' if combined else 'two-pass filter then extract'}")
    print(f"   Categories: {len(categories)}")

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
        
        _finish(metrics, pipeline, {'results': {k: v for k, v in results.items()
                                                if k != 'top_countries'}})
        return results

    except PipelineHalted as halt:
        print(f"\n⏸  Pipeline halted cleanly: {halt.reason}")
        _finish(metrics, pipeline, {'halted': True, 'reason': halt.reason})
        sys.exit(EXIT_QUOTA_EXHAUSTED)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        _finish(metrics, pipeline, {'failed': True, 'error': str(e)})
        return None

def run_collection_only(args):
    """Run only the data collection stage"""
    print("📥 Starting data collection only")
    
    metrics = _make_metrics(args, 'collect', False)
    pipeline = UniversityExtractionPipeline(args.database, metrics=metrics)
    categories = UNIVERSITY_CATEGORIES
    if args.benchmark:
        categories = BENCHMARK_CATEGORIES
    elif args.categories:
        categories = args.categories.split(',')

    pages_collected = pipeline.collect_raw_data(categories, skip_completed=args.skip_completed)
    print(f"✅ Collected {pages_collected} pages")
    _finish(metrics, pipeline, {'pages_collected': pages_collected})
    return pages_collected

def run_filtering_only(args):
    """Run only the university filtering stage"""
    print("🔍 Starting university filtering only")
    
    combined = _combined_flag(args)
    metrics = _make_metrics(args, 'combined' if combined else 'filter', combined)
    pipeline = UniversityExtractionPipeline(args.database, metrics=metrics,
                                            combined=combined)
    try:
        if combined:
            res = pipeline.filter_and_extract_combined(args.batch_size)
            print(f"✅ {res['universities_filtered']} universities, "
                  f"{res['universities_extracted']} extracted")
            _finish(metrics, pipeline, {'results': res})
            return res
        universities_found = pipeline.filter_universities(args.batch_size)
        print(f"✅ Found {universities_found} universities")
        _finish(metrics, pipeline, {'universities_found': universities_found})
        return universities_found
    except PipelineHalted as halt:
        print(f"\n⏸  {halt.reason}")
        _finish(metrics, pipeline, {'halted': True, 'reason': halt.reason})
        sys.exit(EXIT_QUOTA_EXHAUSTED)

def run_extraction_only(args):
    """Run only the data extraction stage"""
    print("📊 Starting data extraction only")
    
    metrics = _make_metrics(args, 'extract', False)
    pipeline = UniversityExtractionPipeline(args.database, metrics=metrics)
    try:
        universities_extracted = pipeline.extract_university_details()
        print(f"✅ Extracted {universities_extracted} university details")
        _finish(metrics, pipeline, {'universities_extracted': universities_extracted})
        return universities_extracted
    except PipelineHalted as halt:
        print(f"\n⏸  {halt.reason}")
        _finish(metrics, pipeline, {'halted': True, 'reason': halt.reason})
        sys.exit(EXIT_QUOTA_EXHAUSTED)

def run_deduplication(args):
    """Run deduplication process"""
    print("🔄 Starting deduplication")
    
    metrics = _make_metrics(args, 'dedupe', False)
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
        if metrics:
            with metrics.stage('dedupe'):
                results = deduplicator.deduplicate_all()
                metrics.record_items(results.get('initial_count', 0))
        else:
            results = deduplicator.deduplicate_all()
        print(f"✅ Deduplication completed:")
        print(f"   • Initial count: {results['initial_count']}")
        print(f"   • Final count: {results['final_count']}")
        print(f"   • Merged duplicates: {results['merged_count']}")
        _finish(metrics, None, {'results': results})
        return results

def show_statistics(args):
    """Show current database statistics"""
    print("📈 Database Statistics")
    print("=" * 30)
    
    db = UniversityDatabase(args.database)
    stats = db.get_statistics()
    
    print(f"Total pages:            {stats.get('total_pages', 0)}")
    print(f"Processed pages:        {stats.get('processed_pages', 0)}")
    print(f"Universities (rows):    {stats.get('universities', 0)}")
    print(f"Classified universities:{stats.get('classified_universities', 0)}")
    print(f"Pending classification: {stats.get('pending_classification', 0)}")
    print(f"Pending extraction:     {stats.get('pending_extraction', 0)}")
    print(f"Abandoned (max retries):{stats.get('abandoned_pages', 0)}")
    
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
    
    combined = _combined_flag(args)
    metrics = _make_metrics(args, 'resume', combined)
    pipeline = UniversityExtractionPipeline(args.database, metrics=metrics,
                                            combined=combined)
    try:
        results = pipeline.resume_pipeline()
    except PipelineHalted as halt:
        print(f"\n⏸  {halt.reason}")
        _finish(metrics, pipeline, {'halted': True, 'reason': halt.reason})
        sys.exit(EXIT_QUOTA_EXHAUSTED)
    _finish(metrics, pipeline, {'results': {k: v for k, v in results.items()
                                            if k != 'top_countries'}})

    print("✅ Pipeline resumed successfully")
    print(f"📊 Current status:")
    for key, value in results.items():
        print(f"   • {key}: {value}")
    
    return results

def probe_limits(args):
    """Discover the key's real rate limits by reading them off a live 429.

    Gemini reports the violated quota's id and value in the error body, so a
    short burst is enough to learn the true RPM without guessing from docs.
    This spends real quota -- each request counts against the daily budget.
    """
    from concurrent.futures import ThreadPoolExecutor
    from google import genai
    from google.genai import types
    from src.gemini_client import extract_quota_violations, parse_retry_after, is_rate_limit_error
    from config import GEMINI_API_KEY

    print(f"🔬 Probing limits for model {GEMINI_MODEL} "
          f"({args.max_requests} tiny requests)...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level='minimal'))

    def one(_):
        try:
            client.models.generate_content(model=GEMINI_MODEL, contents='Say: A', config=cfg)
            return None
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=args.max_requests) as pool:
        errors = [e for e in pool.map(one, range(args.max_requests)) if e is not None]

    succeeded = args.max_requests - len(errors)
    print(f"   succeeded: {succeeded}/{args.max_requests}")

    found = {}
    for exc in errors:
        if is_rate_limit_error(exc):
            for qid, qval in extract_quota_violations(exc):
                found[qid] = qval
            retry_after = parse_retry_after(exc)
            if retry_after:
                print(f"   server Retry-After: {retry_after:.1f}s")
            break

    if found:
        print("\n   Quotas reported by the API:")
        for qid, qval in found.items():
            print(f"     {qid} = {qval}")
        print("\n   Set the matching GEMINI_RPM / GEMINI_RPD values in .env.")
    else:
        print("   No rate limit hit -- the limit is above this burst size.")
    return found


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="University Catalog Builder")
    parser.add_argument('--database', default=DATABASE_PATH, help='Database file path')
    parser.add_argument('--log-level', default=LOG_LEVEL, help='Logging level')
    parser.add_argument('--metrics-dir', default=METRICS_DIR,
                        help='Directory for run metrics JSON files')
    parser.add_argument('--run-label', help='Label for this run in the metrics file')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Disable metrics collection')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    def add_ab_flags(sp):
        """Flags that select between the two Gemini call paths."""
        grp = sp.add_mutually_exclusive_group()
        grp.add_argument('--combined', action='store_true',
                         help='Merge filter+extract into ONE Gemini call per page batch')
        grp.add_argument('--two-pass', action='store_true',
                         help='Force the separate filter-then-extract path')
    
    # Full pipeline
    full_parser = subparsers.add_parser('run', help='Run complete pipeline')
    full_parser.add_argument('--categories', help='Comma-separated list of categories')
    full_parser.add_argument('--skip-dedup', action='store_true', help='Skip deduplication')
    full_parser.add_argument('--benchmark', action='store_true',
                             help='Use the fixed BENCHMARK_CATEGORIES subset')
    add_ab_flags(full_parser)

    # Individual stages
    collect_parser = subparsers.add_parser('collect', help='Run data collection only')
    collect_parser.add_argument('--categories', help='Comma-separated list of categories')
    collect_parser.add_argument('--benchmark', action='store_true',
                                help='Use the fixed BENCHMARK_CATEGORIES subset')
    collect_parser.add_argument('--skip-completed', action='store_true',
                                help='Skip categories already collected')

    filter_parser = subparsers.add_parser('filter', help='Run university filtering only')
    filter_parser.add_argument('--batch-size', type=int, default=None,
                               help='Pages per Gemini request (default: from config)')
    add_ab_flags(filter_parser)

    extract_parser = subparsers.add_parser('extract', help='Run data extraction only')
    
    # Deduplication
    dedup_parser = subparsers.add_parser('dedupe', help='Run deduplication')
    dedup_parser.add_argument('--report-only', action='store_true', help='Generate report only')
    
    # Utilities
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    export_parser = subparsers.add_parser('export', help='Export data')
    export_parser.add_argument('output', help='Output file path')
    
    resume_parser = subparsers.add_parser('resume', help='Resume pipeline')
    add_ab_flags(resume_parser)

    limits_parser = subparsers.add_parser(
        'probe-limits', help='Ask the live API what rate limits this key actually has')
    limits_parser.add_argument('--max-requests', type=int, default=12,
                               help='Requests to burst before giving up (spends quota)')

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Ensure directories exist
    db_dir = os.path.dirname(args.database)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
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
    elif args.command == 'probe-limits':
        return probe_limits(args)
    else:
        parser.print_help()
        return None

if __name__ == '__main__':
    main()