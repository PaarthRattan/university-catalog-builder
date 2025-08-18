import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
WIKIPEDIA_USER_AGENT = os.getenv('WIKIPEDIA_USER_AGENT', 'UniversityCatalog/1.0')
WIKIPEDIA_API_URL = 'https://en.wikipedia.org/w/api.php'

# Database Configuration
DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/universities.db')

# Rate Limiting
WIKIPEDIA_REQUESTS_PER_SECOND = 10
GEMINI_REQUESTS_PER_MINUTE = 60

# Processing Configuration
BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 1

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# University Categories to Search
UNIVERSITY_CATEGORIES = [
    'Category:Universities by country',
    'Category:Educational institutions by country',
    'Category:Universities and colleges',
    'Category:Higher education institutions',
    'Category:Public universities',
    'Category:Private universities',
    'Category:Technical universities',
    'Category:Medical schools',
    'Category:Business schools',
    'Category:Art schools'
]