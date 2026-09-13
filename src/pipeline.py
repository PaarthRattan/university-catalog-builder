import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

from .wikipedia_client import WikipediaClient
from .gemini_client import GeminiClient, GeminiCallError
from .rate_limiter import DailyQuotaExhausted
from .database import UniversityDatabase
from config import (
    UNIVERSITY_CATEGORIES, DATABASE_PATH, GEMINI_MODEL, GEMINI_CLASSIFY_BATCH_SIZE,
    GEMINI_COMBINED_BATCH_SIZE, GEMINI_COMBINED_FILTER_EXTRACT,
    UNIVERSITY_CONFIDENCE_THRESHOLD, MAX_PAGE_ATTEMPTS,
)

logger = logging.getLogger(__name__)


class PipelineHalted(Exception):
    """Raised when the pipeline stops cleanly at a hard quota boundary.

    Distinct from a crash: all work up to this point is committed and the run can
    be resumed with `python main.py resume` once quota resets.
    """

    def __init__(self, reason: str, stats: Dict = None):
        super().__init__(reason)
        self.reason = reason
        self.stats = stats or {}


class UniversityExtractionPipeline:
    def __init__(self, db_path: str = DATABASE_PATH, metrics=None,
                 combined: bool = None):
        self.db = UniversityDatabase(db_path)
        self.metrics = metrics
        # RPD state is keyed by MODEL, not by database, and lives alongside the
        # data directory so it survives process restarts.
        #
        # The quota's own scope is what matters here: the server reports it as
        # GenerateRequestsPerDayPerProjectPerModel, i.e. per API project per
        # model. Keying the state file by database path instead would split the
        # counter across databases that actually share one server-side budget,
        # so two pipelines on one key would each believe they had a full day's
        # quota and both walk into 429s.
        import os as _os
        state_dir = _os.path.dirname(db_path) or "."
        state_path = _os.path.join(state_dir, f".ratelimit_{GEMINI_MODEL}.json")
        self.wikipedia = WikipediaClient()
        self.gemini = GeminiClient(metrics=metrics, state_path=state_path)
        self.combined = (GEMINI_COMBINED_FILTER_EXTRACT if combined is None else combined)

    # ------------------------------------------------------------- utilities

    def _stage(self, name: str):
        """Context manager for a metrics stage; a no-op when metrics are off."""
        if self.metrics:
            return self.metrics.stage(name)
        from contextlib import nullcontext
        return nullcontext()

    def _count_items(self, n: int):
        if self.metrics:
            self.metrics.record_items(n)

    def _halt_on_quota(self, exc: DailyQuotaExhausted, stage: str) -> PipelineHalted:
        """Turn daily-quota exhaustion into a clean, resumable stop."""
        stats = self.db.get_statistics()
        violations = getattr(exc, "violations", None)
        detail = (f" Server reported: {violations}." if violations else "")
        server_msg = getattr(exc, "server_message", None)
        if server_msg:
            detail += f" Message: {server_msg}"
        msg = (f"Daily Gemini quota exhausted during {stage}. {exc}.{detail} "
               f"All completed work is checkpointed; "
               f"resume with `python main.py resume` after the quota window rolls.")
        logger.warning(msg)
        self.db.log_processing_stage(stage, "halted_quota", msg)
        if self.metrics:
            self.metrics.add_note(msg)
        return PipelineHalted(msg, stats)

    # --------------------------------------------------------------- stage 1

    def collect_raw_data(self, categories: List[str] = None,
                         skip_completed: bool = False) -> int:
        """Stage 1: Collect raw Wikipedia pages from university categories."""
        if not categories:
            categories = UNIVERSITY_CATEGORIES

        with self._stage("collect"):
            already = self.db.get_collected_categories() if skip_completed else set()
            todo = [c for c in categories if c not in already]
            if already:
                logger.info("Skipping %d already-collected categories", len(categories) - len(todo))

            logger.info("Starting data collection from %d categories", len(todo))
            self.db.log_processing_stage("data_collection", "started",
                                         f"Processing {len(todo)} categories")

            all_pages = set()
            for category in tqdm(todo, desc="Processing categories"):
                try:
                    pages = self.wikipedia.get_category_members(category, limit=1000)
                    for subcat in self.wikipedia.get_subcategories(category)[:5]:
                        pages.extend(self.wikipedia.get_category_members(subcat, limit=500))
                    for page in pages:
                        all_pages.add((page['page_id'], page['title'], page['url']))
                    self.db.mark_category_collected(category, len(pages))
                    logger.info("Category %s: %d pages found", category, len(pages))
                except Exception as exc:
                    logger.error("Error processing category %s: %s", category, exc)
                    continue

            unique_pages = [{'page_id': pid, 'title': t, 'url': u}
                            for pid, t, u in all_pages]
            logger.info("Found %d unique pages, getting extracts...", len(unique_pages))

            extracts = self.wikipedia.get_page_extracts(
                [p['page_id'] for p in unique_pages])

            total_pages = 0
            for page in tqdm(unique_pages, desc="Storing raw pages"):
                info = extracts.get(page['page_id'], {})
                if self.db.insert_raw_page({
                    'page_id': page['page_id'],
                    'title': page['title'],
                    'url': page['url'],
                    'extract': info.get('extract', ''),
                    'categories': info.get('categories', []),
                }):
                    total_pages += 1

            self._count_items(total_pages)
            self.db.log_processing_stage("data_collection", "completed",
                                         f"Collected {total_pages} pages")
            logger.info("Data collection completed: %d pages stored", total_pages)
            return total_pages

    # --------------------------------------------------------------- stage 2

    def filter_universities(self, batch_size: int = None) -> int:
        """Stage 2: classify pages with Gemini, batched and concurrent.

        batch_size is pages-per-Gemini-request. Concurrency is bounded by the
        client's semaphore, which is itself capped at the RPM limit.
        """
        batch_size = batch_size or GEMINI_CLASSIFY_BATCH_SIZE
        workers = self.gemini.max_concurrency

        with self._stage("filter"):
            logger.info("Starting university filtering (batch=%d, workers=%d)",
                        batch_size, workers)
            self.db.log_processing_stage("university_filtering", "started")

            universities_found = 0
            processed_count = 0
            halted = None

            while halted is None:
                # Pull enough work to keep every worker busy for one round.
                pending = self.db.get_unprocessed_pages(
                    batch_size * workers, max_attempts=MAX_PAGE_ATTEMPTS)
                if not pending:
                    break

                chunks = [pending[i:i + batch_size]
                          for i in range(0, len(pending), batch_size)]

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(self.gemini.batch_university_classification,
                                           chunk, batch_size): chunk
                               for chunk in chunks}
                    for future in as_completed(futures):
                        chunk = futures[future]
                        try:
                            results = future.result()
                        except DailyQuotaExhausted as exc:
                            halted = self._halt_on_quota(exc, "university_filtering")
                            for page in chunk:
                                self.db.record_failed_attempt(page['page_id'])
                            continue
                        except Exception as exc:
                            logger.error("Classification chunk failed: %s", exc)
                            for page in chunk:
                                self.db.record_failed_attempt(page['page_id'])
                            continue

                        for res in results:
                            page = res['page']
                            if not res.get('ok'):
                                # A failure is NOT a negative classification.
                                self.db.record_failed_attempt(page['page_id'])
                                logger.debug("Page %s unresolved: %s",
                                             page['title'], res.get('error'))
                                continue
                            is_uni = (res['is_university'] and
                                      res['confidence'] >= UNIVERSITY_CONFIDENCE_THRESHOLD)
                            self.db.mark_page_processed(page['page_id'], is_uni,
                                                        res['confidence'])
                            processed_count += 1
                            if is_uni:
                                universities_found += 1

                self._count_items(len(pending))

            self.db.log_processing_stage(
                "university_filtering", "halted_quota" if halted else "completed",
                f"Found {universities_found} universities from {processed_count} pages")
            logger.info("Filtering done: %d universities from %d pages",
                        universities_found, processed_count)
            if halted:
                raise halted
            return universities_found

    # --------------------------------------------------------------- stage 3

    def extract_university_details(self) -> int:
        """Stage 3: extract structured details, concurrent and checkpointed."""
        workers = self.gemini.max_concurrency

        with self._stage("extract"):
            pages = self.db.get_pages_for_extraction(
                min_confidence=UNIVERSITY_CONFIDENCE_THRESHOLD)
            logger.info("Extracting details for %d universities (workers=%d)",
                        len(pages), workers)
            self.db.log_processing_stage("data_extraction", "started")

            extracted_count = 0
            halted = None

            def work(page):
                return page, self.gemini.extract_university_data(
                    page['title'], page['extract'])

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(work, p) for p in pages]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Extracting university details"):
                    try:
                        page, data = future.result()
                    except DailyQuotaExhausted as exc:
                        if halted is None:
                            halted = self._halt_on_quota(exc, "data_extraction")
                        continue
                    except GeminiCallError as exc:
                        logger.error("Extraction failed: %s", exc)
                        continue
                    except Exception as exc:
                        logger.error("Unexpected extraction error: %s", exc)
                        continue

                    if self._store_university(page, data, page.get('confidence')):
                        extracted_count += 1

            self._count_items(extracted_count)
            self.db.log_processing_stage(
                "data_extraction", "halted_quota" if halted else "completed",
                f"Extracted details for {extracted_count} universities")
            logger.info("Extraction completed: %d universities", extracted_count)
            if halted:
                raise halted
            return extracted_count

    def _store_university(self, page, data: Dict, confidence: float = None) -> bool:
        ok = self.db.insert_university({
            'name': data.get('name') or page['title'],
            'wikipedia_url': page['url'],
            'city': data.get('city'),
            'country': data.get('country'),
            'founded_year': data.get('founded'),
            'closed_year': data.get('closed'),
            'university_type': data.get('type'),
            'confidence_score': confidence if confidence is not None else 1.0,
        })
        if ok:
            self.db.mark_page_extracted(page['page_id'])
        return ok

    # ----------------------------------------------- combined filter+extract

    def filter_and_extract_combined(self, batch_size: int = None) -> Dict:
        """Stages 2+3 merged into ONE Gemini call per batch (A/B path).

        Enabled by GEMINI_COMBINED_FILTER_EXTRACT. Produces the same rows in the
        same tables as the two-pass path.
        """
        batch_size = batch_size or GEMINI_COMBINED_BATCH_SIZE
        workers = self.gemini.max_concurrency

        with self._stage("filter_extract_combined"):
            logger.info("Combined classify+extract (batch=%d, workers=%d)",
                        batch_size, workers)
            self.db.log_processing_stage("combined_filter_extract", "started")

            universities_found = 0
            extracted_count = 0
            processed_count = 0
            halted = None

            while halted is None:
                pending = self.db.get_unprocessed_pages(
                    batch_size * workers, max_attempts=MAX_PAGE_ATTEMPTS)
                if not pending:
                    break

                chunks = [pending[i:i + batch_size]
                          for i in range(0, len(pending), batch_size)]

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(self.gemini.classify_and_extract,
                                           chunk, batch_size): chunk
                               for chunk in chunks}
                    for future in as_completed(futures):
                        chunk = futures[future]
                        try:
                            results = future.result()
                        except DailyQuotaExhausted as exc:
                            halted = self._halt_on_quota(exc, "combined_filter_extract")
                            for page in chunk:
                                self.db.record_failed_attempt(page['page_id'])
                            continue
                        except Exception as exc:
                            logger.error("Combined chunk failed: %s", exc)
                            for page in chunk:
                                self.db.record_failed_attempt(page['page_id'])
                            continue

                        for res in results:
                            page = res['page']
                            if not res.get('ok'):
                                self.db.record_failed_attempt(page['page_id'])
                                continue
                            is_uni = (res['is_university'] and
                                      res['confidence'] >= UNIVERSITY_CONFIDENCE_THRESHOLD)
                            self.db.mark_page_processed(page['page_id'], is_uni,
                                                        res['confidence'])
                            processed_count += 1
                            if is_uni:
                                universities_found += 1
                                if res.get('details') and self._store_university(
                                        page, res['details'], res['confidence']):
                                    extracted_count += 1

                self._count_items(len(pending))

            self.db.log_processing_stage(
                "combined_filter_extract",
                "halted_quota" if halted else "completed",
                f"{universities_found} universities, {extracted_count} extracted, "
                f"from {processed_count} pages")
            if halted:
                raise halted
            return {'universities_filtered': universities_found,
                    'universities_extracted': extracted_count,
                    'pages_processed': processed_count}

    # ------------------------------------------------------------------ runs

    def run_full_pipeline(self, categories: List[str] = None) -> Dict:
        start_time = time.time()
        logger.info("Starting full pipeline (combined=%s)", self.combined)
        results = {}

        try:
            results['pages_collected'] = self.collect_raw_data(categories)
            if self.combined:
                results.update(self.filter_and_extract_combined())
            else:
                results['universities_filtered'] = self.filter_universities()
                results['universities_extracted'] = self.extract_university_details()

            results.update(self.db.get_statistics())
            results['total_time_seconds'] = time.time() - start_time
            self.db.log_processing_stage("full_pipeline", "completed",
                                         "Pipeline completed successfully",
                                         results.get('universities_extracted'))
            logger.info("Pipeline completed in %.2fs", results['total_time_seconds'])
        except PipelineHalted:
            raise
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)
            self.db.log_processing_stage("full_pipeline", "failed", str(exc))
            raise

        return results

    def get_statistics(self) -> Dict:
        return self.db.get_statistics()

    def resume_pipeline(self) -> Dict:
        """Resume from wherever the last run stopped.

        Driven by the actual outstanding work counts rather than by the previous
        implementation's `universities == 0` test, which reported a partially
        extracted run as complete and did nothing.
        """
        stats = self.db.get_statistics()
        logger.info("Resume: %d pending classification, %d pending extraction, "
                    "%d abandoned", stats['pending_classification'],
                    stats['pending_extraction'], stats['abandoned_pages'])

        if stats['total_pages'] == 0:
            logger.warning("No pages collected yet -- run `collect` first.")
            return self.get_statistics()

        did_work = False
        if stats['pending_classification'] > 0:
            logger.info("Resuming classification of %d pages",
                        stats['pending_classification'])
            if self.combined:
                self.filter_and_extract_combined()
            else:
                self.filter_universities()
            did_work = True

        if not self.combined:
            remaining = self.db.get_statistics()['pending_extraction']
            if remaining > 0:
                logger.info("Resuming extraction of %d universities", remaining)
                self.extract_university_details()
                did_work = True

        if not did_work:
            logger.info("Nothing outstanding -- pipeline is complete.")
        return self.get_statistics()
