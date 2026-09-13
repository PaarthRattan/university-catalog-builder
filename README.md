# University Catalog Builder

A comprehensive system for extracting, cataloging, and analyzing universities from Wikipedia using Google Gemini AI. This project builds a structured database of universities worldwide with temporal data suitable for per-capita analysis.

## Features

- **Automated Data Collection**: Extracts university data from Wikipedia categories and search results
- **AI-Powered Filtering**: Uses Google Gemini to identify genuine universities vs. other institutions
- **Structured Data Extraction**: Extracts founding years, locations, types, and other metadata
- **Deduplication**: Intelligent merging of duplicate entries using fuzzy matching
- **Temporal Analysis**: Track university counts over time for historical analysis
- **Export Capabilities**: Multiple output formats (CSV, JSON, Excel)
- **Comprehensive Analytics**: Per-capita analysis and statistical reports

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: GEMINI_API_KEY
```

### 2. Run the Pipeline

```bash
# Run complete pipeline
python main.py run

# Or run individual stages
python main.py collect    # Data collection only
python main.py filter     # University filtering only  
python main.py extract    # Data extraction only
python main.py dedupe     # Deduplication only
```

### 3. Export Data

```bash
# Export to CSV
python main.py export universities.csv

# Export to JSON
python main.py export universities.json

# View statistics
python main.py stats
```

## Architecture

### Data Pipeline

```
Wikipedia Categories → Raw Pages → Gemini Filtering → 
Data Extraction → Deduplication → Analysis
```

### Key Components

- **Wikipedia Client** (`src/wikipedia_client.py`): Handles API requests and data collection
- **Gemini Client** (`src/gemini_client.py`): AI-powered classification and extraction
- **Pipeline** (`src/pipeline.py`): Orchestrates the complete workflow
- **Database** (`src/database.py`): SQLite storage with structured schema
- **Deduplication** (`src/deduplication.py`): Fuzzy matching and merging
- **Analytics** (`src/analytics.py`): Statistical analysis and reporting

## Configuration

### Categories Searched

The system searches these Wikipedia categories by default:

- Universities by country
- Educational institutions by country  
- Universities and colleges
- Higher education institutions
- Public/Private/Technical universities
- Medical schools, Business schools, Art schools

### Rate Limiting

Gemini RPM, TPM and RPD are enforced client-side, all three independently;
exceeding any one is treated as a 429 before the request is sent. Configure in
`.env`:

```
GEMINI_RPM=5          # measured free-tier limit
GEMINI_TPM=250000
GEMINI_RPD=20         # measured free-tier limit
GEMINI_CLASSIFY_BATCH_SIZE=10
GEMINI_MAX_CONCURRENCY=8          # capped at GEMINI_RPM in code
GEMINI_COMBINED_FILTER_EXTRACT=0  # A/B flag for the merged path
```

Run `python main.py probe-limits` to read the real quota values off the live
API rather than trusting these defaults. The daily budget persists across runs;
on exhaustion the pipeline checkpoints and exits with code 2, and
`python main.py resume` continues from that point.

- Wikipedia API: 10 requests/second, bounded concurrency

## Database Schema

### Universities Table
- `name`: University name
- `wikipedia_url`: Source Wikipedia URL
- `city`, `country`: Geographic location
- `founded_year`, `closed_year`: Temporal data
- `university_type`: public/private/other
- `confidence_score`: AI classification confidence

### Raw Pages Table
- Stores all collected Wikipedia pages
- Tracks processing status
- Preserves original content for reprocessing

## Analytics Features

### Temporal Analysis
- Universities per capita by country/year
- Founding timeline by decade
- Historical growth trends

### Geographic Distribution
- Country and city statistics
- Regional analysis
- Coverage assessment

### Data Quality Metrics
- Missing data analysis
- Duplicate detection reports
- Confidence score distributions

## Usage Examples

### Basic Pipeline
```bash
# Run with specific categories
python main.py run --categories "Category:Universities in the United States,Category:Universities in Canada"

# Skip deduplication
python main.py run --skip-dedup
```

### Advanced Analysis
```python
from src.analytics import UniversityAnalytics

analytics = UniversityAnalytics('data/universities.db')

# Get universities per capita for 2020
per_capita = analytics.get_universities_per_capita_by_year(2020, population_data)

# Generate comprehensive report
report = analytics.generate_comprehensive_report('analysis_report.json')
```

### Resume Pipeline
```bash
# Resume from last checkpoint
python main.py resume
```

## API Requirements

### Google Gemini API
- Get API key from Google AI Studio
- Set in environment: `GEMINI_API_KEY=your_key_here`
- Used for university classification and data extraction

### Wikipedia API
- No API key required
- Respectful rate limiting implemented
- User-Agent header configured

## Output Formats

### CSV Export
Columns: name, city, country, founded_year, closed_year, university_type, wikipedia_url

### JSON Export
Structured data with nested objects for complex fields

### Analysis Reports
- Comprehensive statistics
- Geographic distribution
- Temporal trends
- Data quality assessment

## Performance

All figures below are measured on real runs. Nothing here is estimated or
extrapolated. Raw metrics files for every run are committed under `metrics/`.

**Measurement environment**

| | |
|---|---|
| Hardware | Apple M3 Pro, 18 GB RAM, macOS 14.4 |
| Date | 2026-09-13 |
| API tier | Google Gemini **free tier** |
| Models | `gemini-3.5-flash-lite` (A/B), `gemini-3.1-flash-lite` (path comparison) |
| Sample | 16 pages from `Category:Universities and colleges in Delaware` + `...in Rhode Island` |

### Measured rate limits (free tier)

These were read off live 429 responses, not from documentation. Reproduce with
`python main.py probe-limits`.

| Limit | Value | Source |
|---|---|---|
| Requests/minute | **5** | `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`, quotaValue `5` |
| Requests/day | **20** | `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quotaValue `20` |
| Tokens/minute | unverified | never hit; unreachable at 5 RPM |

Quotas are **per model per project**, so each model carries its own 20/day budget.

### Batching + concurrency: before/after

Identical 16-page sample, identical model, identical prompt. "Before" is one
page per Gemini request, serial; "after" is 10 pages per request, 5 concurrent.

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Gemini requests | 16 | 2 | **8× fewer** |
| Wall clock | 180.7 s | 1.3 s | **140× faster** |
| Pages/min | 5.3 | 744.9 | **140×** |
| Tokens used | 5,200 | 3,165 | 39% fewer |
| Time blocked on rate limits | 170.5 s (94%) | 0.0 s | eliminated |
| Server 429s | 0 | 0 | — |

The before run spent **94% of its wall clock waiting on the rate limiter**, which
is the whole story: at 5 RPM, request *count* is the only thing that matters.
Batching removes the waiting entirely for a sample this size.

**Accuracy is not identical.** On the same 16 pages the two runs agreed on
15/16 classifications (93.8%). The single disagreement was
"List of colleges and universities in Rhode Island" — a list article, which the
unbatched run rejected and the batched run accepted. Note the unbatched run was
already self-inconsistent here, accepting the equivalent Delaware list article,
so this looks like model nondeterminism on an ambiguous input rather than a
batching regression. It is reported rather than smoothed over. Both paths
misclassify list articles, which is a prompt defect independent of batching.

### Two-pass vs combined filter+extract

Same 10 pages, same model (`gemini-3.1-flash-lite`). The combined path is behind
`GEMINI_COMBINED_FILTER_EXTRACT` / `--combined`.

| Metric | Two-pass | Combined | Delta |
|---|---:|---:|---:|
| Gemini requests | 11 | 2 | **5.5× fewer** |
| Wall clock | 121.1 s | 2.2 s | **54× faster** |
| Tokens used | 6,757 | 3,820 | 43% fewer |
| Universities written | 10 | 9 | −1 |
| Rows identical to two-pass | — | 7/10 | — |

**The combined path does not produce identical output**, which is why it is a
flag and not the default. It dropped one record (Naval War College) and differed
on `university_type` for two others. On those two it was arguably more correct
— it labelled Delaware State University and the University of Delaware `public`,
where the two-pass path emitted the unhelpful
`"privately governed, state-assisted"`. Choose the path deliberately.

### What this means at free-tier quota

At 20 requests/day, request count is the entire budget:

| Approach | Pages classifiable per day |
|---|---:|
| One page per request | 20 |
| 10 pages per request | 200 |
| Combined, 5 pages per request | 100 (classified **and** extracted) |

## Results

Scope note: these are the runs that were actually executed. A full run over the
complete `UNIVERSITY_CATEGORIES` list was **not** performed — see below.

| Run | Pages | Universities | Gemini requests | Wall clock | Cost |
|---|---:|---:|---:|---:|---:|
| Collection (2 categories) | 24 | — | 0 | 4.7 s | $0.00 |
| Classification A/B (before) | 16 | 15 | 16 | 180.7 s | $0.00 |
| Classification A/B (after) | 16 | 16 | 2 | 1.3 s | $0.00 |
| Full two-pass (10 pages) | 10 | 10 | 11 | 121.1 s | $0.00 |
| Full combined (10 pages) | 10 | 9 | 2 | 2.2 s | $0.00 |

Cost is $0.00 because every run was on the free tier, which bills nothing. Set
`GEMINI_PRICE_INPUT_PER_MTOK` / `GEMINI_PRICE_OUTPUT_PER_MTOK` to get a real
dollar figure on a paid tier; the metrics reporter multiplies measured token
counts by those prices and otherwise reports `$0.0000 (free tier)` rather than
inventing a number.

### The full-catalogue run was not possible on this tier

The free tier allows **20 Gemini requests per day per model**. The complete
category list resolves to thousands of Wikipedia pages. Even at the best
measured density — the combined path at 5 pages per request — that is hundreds
of requests, i.e. **weeks of wall-clock time spanning dozens of daily quota
windows**. It was therefore not attempted, and no figures for it are published
here. Reproducing it needs a paid tier.

The pipeline is built to survive that scenario rather than pretend it away: the
daily quota is tracked across process restarts, exhaustion checkpoints and exits
with code 2 rather than burning retries, and `python main.py resume` picks up
exactly where it stopped.

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Increase delays in config.py
2. **Missing Data**: Check Wikipedia category names
3. **Gemini Errors**: Verify API key and quota
4. **Database Locks**: Close other connections

### Logs
Check `logs/` directory for detailed processing logs

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

MIT License - see LICENSE file for details

## Citation

If using this data for research, please cite:
```
University Catalog Builder (2026)
GitHub: https://github.com/paarthrattan/university-catalog-builder
```