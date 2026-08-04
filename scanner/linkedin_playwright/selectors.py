"""Every LinkedIn CSS selector and scrape-tuning constant, in one place.

LinkedIn's markup uses obfuscated, frequently-changing class names — selector
fragility is this project's single biggest maintenance cost. Collecting them
here means a LinkedIn redesign has exactly one file to fix, instead of a hunt
through 1,500 lines of scraping logic.
"""
_DESC_SELECTORS = (
    "#job-details",
    ".jobs-description__content",
    ".jobs-description-content__text",
    ".jobs-description-content__text--stretch",
    ".jobs-box__html-content",
    "article.jobs-description",
    "section[class*='description']",
    "div[class*='description-content']",
)

_DATE_SELECTORS = (
    ".jobs-unified-top-card__posted-date",
    ".tvm__text--positive",
    "span[class*='posted']",
    "span[class*='date']",
)

# Selectors for the right-side detail panel that appears when clicking a job card
_PANEL_SELECTORS = (
    ".jobs-search__job-details--wrapper",
    ".jobs-search-two-pane__detail-view",
    ".scaffold-layout__detail",
    ".jobs-details",
)


# ── Browser helpers ────────────────────────────────────────────────────────────

_JOBS_PER_PAGE = 25  # LinkedIn's fixed page size for job search
_MIN_PAGES     = 3   # always scan at least this many pages per keyword
_MAX_PAGES     = 4   # LinkedIn search results get unreliable/rate-limited beyond this
_MAX_SCROLL_ATTEMPTS = 8  # cards load ~7 at a time; enough headroom to reach 25 per page

_JOB_DESCRIPTION_PAGE_SETTLE_MS = 750

SEARCH_URL           = "https://www.linkedin.com/jobs/search/"
SEMANTIC_SEARCH_URL  = "https://www.linkedin.com/jobs/search-results/"
