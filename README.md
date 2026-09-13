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

- Wikipedia API: 10 requests/second
- Gemini API: 60 requests/minute
- Configurable batch sizes for optimization

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

### Typical Processing Times
- Data Collection: ~2-5 minutes per 1000 pages
- AI Filtering: ~1-2 minutes per 100 pages  
- Data Extraction: ~2-3 minutes per 100 universities
- Deduplication: ~30 seconds per 1000 universities

### Resource Usage
- Memory: ~100-500MB during processing
- Storage: ~50MB per 10,000 universities
- API Costs: ~$1-5 per 10,000 pages (Gemini)

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