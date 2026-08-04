# Job Scanner

Scrapes job postings from LinkedIn, Naukri, company ATS boards (Greenhouse/Lever/Ashby), and JSearch (an aggregator covering LinkedIn/Indeed/Glassdoor/ZipRecruiter), stores them in a local SQLite database, scores them against your resume with an LLM, and surfaces them through a Streamlit web UI where you can filter, search, track application status, find referral contacts and message them on LinkedIn, and auto-fill job applications.

---

## Features

- **Multi-source scanning** — LinkedIn (via [python-jobspy](https://github.com/Bunsly/JobSpy), no login required), LinkedIn via an authenticated Playwright session, Naukri.com, any company's public Greenhouse/Lever/Ashby job board, and JSearch (RapidAPI aggregator)
- **LLM job scoring** — each job is scored against your resume/summary (skills, company fit, remote fit, role fit), with a full breakdown shown in the UI
- **Structured JD/resume scoring (optional)** — `SCORING_MODE=structured` extracts each job description and your resume into typed JSON (must-haves, YOE, seniority, tech stack, work auth, etc.) with a cheap model, then scores JSON against JSON with a separate model instead of re-sending raw text every time. Structured scores are stored in their own columns alongside the default raw-text score, so the two can be compared side by side
- **Referral finder** — searches LinkedIn for 1st/2nd-degree connections at a company, drafts a referral message with an LLM, and can send it as a LinkedIn DM automatically or open it pre-filled for manual review
- **Application auto-fill** — opens a job's application page in a visible browser and best-effort fills name/email/phone/LinkedIn/resume fields from your saved profile; never auto-submits
- **SQLite persistence** — jobs survive across runs; re-scanning a known job never resets your status
- **Duplicate prevention at write time** — the same role posted under multiple sources (e.g. LinkedIn + a company's own Greenhouse board) collapses to one row instead of being flagged after the fact
- **Application tracking** — mark jobs as `new`, `shortlisted`, `saved`, `applied`, or `rejected` directly in the UI
- **Candidate profile** — store your details, resume (PDF/DOCX), multiple saved search profiles, and company boards to scan
- **Streamlit web UI** — filterable, searchable, scorable job table with a detail view per job
- **CLI + cron support** — headless scanning (`main.py`), a daily all-sources scan + scoring job designed for cron (`cron_scan.py`), a description-backfill utility (`backfill_descriptions.py`), a structured-JD-extraction backfill utility (`backfill_jd_extraction.py`), and a standalone scoring utility (`backfill_score_jobs.py`)

---

## Requirements

- Python 3.10+
- No LinkedIn/Naukri account required for the jobspy-based LinkedIn source or the ATS board/JSearch sources — only needed for the authenticated LinkedIn/Naukri sources and referral/apply automation
- An Anthropic (Claude) or Google (Gemini) API key for scoring, resume summaries, and referral drafting (optional — the rest of the app works without one)

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Job_Scanner
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium     # needed for LinkedIn/Naukri login, referrals, and apply auto-fill
```

### 4. Configure

```bash
cp .env.example .env
```

Edit `.env` — see the comments in `.env.example` for the full list. At minimum:

```env
# CLI-only defaults (the web UI ignores these — see below)
SEARCH_KEYWORDS=data engineer
SEARCH_LOCATION=New York, NY
RESULTS_WANTED=25
HOURS_OLD=72

# Optional but recommended: enables scoring, resume summaries, referral drafts
LLM_PROVIDER=claude
CLAUDE_API_KEY=

# Optional: enables the authenticated LinkedIn/Naukri scan buttons + referrals
LINKEDIN_EMAIL=
LINKEDIN_PASSWORD=
NAUKRI_EMAIL=
NAUKRI_PASSWORD=

# Optional: enables the JSearch aggregator source
JSEARCH_API_KEY=
```

> The web UI reads its scan defaults from the **Job Search Criteria** saved in your profile (Profile tab → Search Profiles → Save Profile). The `.env` file's `SEARCH_*` values only affect the CLI (`main.py`) and are the fallback for `cron_scan.py` when no profile has been saved yet.

---

## Running the Web UI

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

### Jobs tab

#### Sidebar — run a scan

| Field | Description |
|---|---|
| Keywords | Comma-separated job titles/skills — each is searched separately and results are combined |
| Location | City, state, or country (e.g. `Remote`, `USA`, `London, UK`) |
| Max results | How many postings to fetch per keyword (5–100) |
| Posted within | Only return postings younger than this window |

Buttons:
- **🚀 Scan All (parallel)** — runs every source below at once (LinkedIn login runs first and alone to avoid concurrent LinkedIn traffic from one account), then scores new jobs
- **🔍 LinkedIn (jobspy)** — no login required
- **🔗 LinkedIn (login)** — requires `LINKEDIN_EMAIL`/`LINKEDIN_PASSWORD`; first run opens a visible browser to log in once, then reuses the saved session
- **💼 Naukri** — requires `NAUKRI_EMAIL`/`NAUKRI_PASSWORD`, same session-reuse behavior
- **🏢 Company Boards** — pulls from every Greenhouse/Lever/Ashby board saved in the Profile tab; ignores "Posted within"
- **🌐 JSearch** — requires `JSEARCH_API_KEY`

Results are saved to the database immediately. Running the same scan again is safe — existing jobs are updated but your status choices are preserved, and cross-source duplicates of the same role are prevented rather than shown twice.

#### Stats row

Shows live counts of **Total / New / Shortlisted / Applied / Rejected** across all jobs in the database.

#### Filters

| Control | Description |
|---|---|
| Search box | Filter by title, company, or location substring |
| Status dropdown | Show only jobs in a given status bucket |
| Remote only | Filter to jobs flagged `is_remote = true` by the source |
| Minimum score | Hide jobs scored below this threshold (unscored jobs are always shown) |

#### Job table & detail view

Click a row to open its detail view: full description, match score breakdown, status editor, **Apply** (opens the application page in a browser and auto-fills what it can), and the **Referrals** section (find contacts, draft a message, send as a LinkedIn DM or open pre-filled).

#### Scoring

**🎯 Score Jobs** scores every job with a description against your saved candidate summary, running in the background with a live progress bar (a second click cancels it). New jobs found by a scan are auto-scored afterward.

---

### Profile tab

#### Candidate Details

Fill in your name, email, phone, LinkedIn URL, current job title, years of experience, and a professional summary (or generate one from your resume with **✨ Generate from Resume**). Click **💾 Save Details** to persist.

#### Resume

Upload a PDF or DOCX file. The file is stored in the database; text is extracted best-effort to auto-generate your professional summary. Previously uploaded resumes remain accessible via **⬇ Download** until replaced.

#### Search Profiles

Save multiple keyword/location/results/hours/remote-only combos. **⬆ Load** pushes a saved profile's values into the sidebar instantly; **✏️** edits it in place.

#### Company Boards

Add companies whose Greenhouse/Lever/Ashby job board you want scanned directly by the "🏢 Company Boards" sidebar button — just a company name, the ATS, and its board token/slug.

---

## Running the CLI

Make sure the virtual environment is active first (`source venv/bin/activate`).

### `main.py` — one-off scan

```bash
# Basic search — print results to terminal
python main.py --keywords "backend engineer" --location "San Francisco, CA"

# Save results to the SQLite database
python main.py --keywords "machine learning engineer" --location "Remote" --db

# Remote jobs only (uses the source's own is_remote flag, not location text)
python main.py --keywords "python developer" --location "USA" --remote-only --db

# Exclude titles you don't want
python main.py --keywords "engineer" --location "USA" --exclude senior staff principal --db

# Save to both the database and a CSV snapshot
python main.py --keywords "data engineer" --location "NYC" --db --save
```

| Flag | Default | Description |
|---|---|---|
| `--keywords` | `.env` / `software engineer` | Search term |
| `--location` | `.env` / `Remote` | Location string |
| `--results` | `.env` / `25` | Max number of results |
| `--hours` | `.env` / `72` | Max age of postings in hours |
| `--remote-only` | off | Keep only jobs where `is_remote = True` |
| `--exclude WORD …` | none | Drop titles containing any of these words |
| `--db` | off | Persist results to `data/jobs.db` |
| `--save` | off | Write a timestamped CSV to `output/` |

### `cron_scan.py` — daily all-sources scan + scoring

Scans every enabled source (LinkedIn jobspy always; LinkedIn login/Naukri only if a session already exists from the web UI; company boards; JSearch if configured), saves new jobs, then scores everything unscored. Never attempts an interactive login — a cron job can't complete a visible-browser/CAPTCHA flow unattended.

```bash
python cron_scan.py                                          # uses saved search criteria (Profile tab)
python cron_scan.py --keywords "backend engineer,staff engineer" --location "USA"
python cron_scan.py --results 25 --hours 72
```

Defaults come from the saved **Job Search Criteria** profile, falling back to built-in defaults if none is saved. Example cron entry:

```
0 7 * * * cd /path/to/Job_Scanner && venv/bin/python cron_scan.py >> logs/cron_scan.log 2>&1
```

### `backfill_descriptions.py` — recover missing descriptions

Some LinkedIn jobs get saved without a description when jobspy's per-job fetch fails transiently. This re-fetches those directly from LinkedIn's public job page and, once recovered, can score them.

```bash
python backfill_descriptions.py            # backfill descriptions only
python backfill_descriptions.py --score    # also score newly-recovered jobs
```

### `backfill_jd_extraction.py` — backfill structured JD extraction

Only relevant if you've enabled `SCORING_MODE=structured` (see `.env.example`). Extracts structured JD JSON (`jobs.jd_extracted`) for jobs that were saved before this feature existed, or before you switched into structured mode — it's a no-op under `SCORING_MODE=raw`, the default. Logs a `[i/total]` progress line per job as it runs. Extraction only — run `backfill_score_jobs.py` afterward to score the newly-extracted jobs.

```bash
python backfill_jd_extraction.py            # extract structured JD data only
python backfill_jd_extraction.py --top 50   # only the top 50 jobs by score (existing jobs are already ordered by score)
python backfill_jd_extraction.py --force    # re-extract every job, even ones already extracted (e.g. after adding a new field)
```

### `backfill_score_jobs.py` — score jobs that need it

Scores every job that needs it (raw or structured, depending on `SCORING_MODE`) without touching JD extraction or description backfill — split out from `backfill_jd_extraction.py` so extraction and scoring can be run/scheduled independently.

```bash
python backfill_score_jobs.py            # score every job that needs it
python backfill_score_jobs.py --top 50   # only the top 50 jobs by score
```

### `backfill_content_hash.py` — backfill dedup hashes

Populates `jobs.content_hash` for rows saved before that column existed. New jobs get one automatically at save time, so this is only needed once (or after restoring an old database). See [Deduplication](#deduplication--upsert-behavior) for what the hash is used for.

```bash
python backfill_content_hash.py            # fill in missing hashes only
python backfill_content_hash.py --force    # recompute every row's hash
```

### `clear_db.py` — reset tables

```bash
python clear_db.py --jobs         # clear the jobs table
python clear_db.py --referrals    # clear the referrals table
python clear_db.py --all          # clear both
```

---

## Testing

```bash
pytest                      # run the suite
ruff check .                # lint
```

The suite runs entirely offline — no network, no browser, no API key. Coverage is concentrated on the pure logic: the ATS/aggregator row mapping (`greenhouse`/`lever`/`ashby`/`jsearch`), the scan kernel in `scanner/search.py`, the shared CLI flags, DataFrame filters, database migrations and dedup ordering, the Playwright launch helper, Naukri's row parsing, and the LLM scoring/extraction/breaker logic.

Two shared test helpers are worth knowing about:

- `tests/conftest.py` — `isolated_db` repoints `database.DB_PATH` at a temp file (tables are created lazily, so no teardown is needed); `stub_fetch_json` stubs the HTTP call the ATS scrapers share.
- `tests/fakes.py` — `FakePage` / `FakeElement` / `FakePlaywright` stand in for Playwright, which is enough to test scraping logic without a browser.

**Not covered:** the live Playwright flows (`linkedin_playwright.py`'s scraping/login, `naukri_playwright.py`'s `search_jobs`/`login`, `apply.py`) and the Streamlit UI — those need a real session, browser, or running server. The LinkedIn scraper in particular depends on obfuscated CSS class names that only a live run can validate; use `debug_linkedin_scan.py` for that.

---

## Project Structure

```
Job_Scanner/
├── app.py                    # Streamlit entry point — wires the sidebar to the two tabs
├── cli_common.py             # Shared argparse flags + logging for the scripts below
├── main.py                   # CLI entry point (one-off scan)
├── cron_scan.py              # Daily all-sources scan + scoring, designed for cron
├── backfill_descriptions.py  # Recover missing LinkedIn descriptions, optionally score
├── backfill_jd_extraction.py # Backfill structured JD extraction (SCORING_MODE=structured only)
├── backfill_score_jobs.py    # Score every job that needs it (raw or structured)
├── backfill_content_hash.py  # Backfill jobs.content_hash for pre-existing rows
├── clear_db.py               # Reset jobs/referrals tables
├── requirements.txt
├── pytest.ini / ruff.toml    # Test + lint configuration
├── .env.example              # Copy to .env — see its comments for the full option list
│
├── scanner/                  # The only layer that touches the network or the database
│   ├── __init__.py           # Package facade — re-exports the public API
│   ├── search.py             # Scan kernel: SearchCriteria, ScanResult, the per-keyword loop
│   ├── linkedin.py           # jobspy wrapper (no login) + description backfill
│   ├── linkedin_playwright.py  # Authenticated LinkedIn scraper, referral finder, DM sender
│   ├── naukri_playwright.py  # Authenticated Naukri scraper
│   ├── greenhouse.py         # Public Greenhouse board API
│   ├── lever.py              # Public Lever board API
│   ├── ashby.py              # Public Ashby board API
│   ├── ats_common.py         # Shared helpers for the three ATS scrapers above
│   ├── ats_registry.py       # Maps the stored `ats` value to its fetcher
│   ├── jsearch.py            # JSearch (RapidAPI) aggregator search
│   ├── manual.py             # Add a single job by pasting its URL
│   ├── browser.py            # Shared Playwright launch config (UA, stealth script, launch args)
│   ├── apply.py              # Application form auto-fill automation
│   ├── llm/                  # All LLM calls
│   │   ├── __init__.py       # Provider selection, circuit breaker, tracing, shared plumbing
│   │   ├── extraction.py     # Resume summary + structured JD/resume extraction models
│   │   ├── raw_scoring.py    # SCORING_MODE=raw — scores raw JD text
│   │   ├── structured_scoring.py  # SCORING_MODE=structured — scores JSON against JSON
│   │   ├── referral.py       # Referral message drafting + apply-form field matching
│   │   └── prompts/          # Prompt text, kept out of the Python source
│   ├── scoring.py            # score_unscored_jobs() + the structured-mode extraction helpers
│   ├── database.py           # SQLite: jobs + referrals — save, query, dedup, status/score updates
│   ├── profile.py            # SQLite: candidate, resume, search_criteria, company_boards
│   ├── filters.py            # DataFrame filter helpers
│   └── config.py             # Reads CLI defaults from .env (main.py only)
│
├── ui/                       # Streamlit UI, split by concern; app.py is only the wiring
│   ├── sidebar.py            # Search settings + the per-source scan buttons
│   ├── jobs_tab.py           # Stats row, filters, job table, row selection
│   ├── detail_panel.py       # Selected-job modal: description, score, status editor
│   ├── profile_tab.py        # Candidate details, resume, search profiles, company boards
│   ├── referrals.py          # Referral contact discovery, drafting, sending
│   ├── scoring.py            # Score button, auto-scoring after a scan, score display
│   ├── scan_handlers.py      # Main-thread scan orchestration (progress bars, "Scan All")
│   ├── scan_runners.py       # Per-source setup; delegates the scan loop to scanner/search.py
│   └── constants.py          # Shared thresholds, poll intervals, log-box sizes
│
├── tests/                    # pytest suite — mocked, no network or browser required
│   ├── conftest.py           # Shared fixtures (isolated_db, stub_fetch_json)
│   └── fakes.py              # Fake Playwright page/element/browser objects
│
├── data/
│   ├── jobs.db                # SQLite database (created on first run)
│   └── playwright_sessions/   # Saved LinkedIn/Naukri login sessions
│
└── output/
    └── *.csv                  # CSV exports (git-ignored)
```

---

## Database Schema

All tables live in `data/jobs.db`.

### `jobs`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | Source-prefixed id (`li-...`, `gh-{token}-...`, `lv-{company}-...`, `ash-{board}-{hash}`, `jsearch-...`, `naukri-...`) |
| `site` | TEXT | Source name (`linkedin`, `greenhouse`, `lever`, `ashby`, `naukri`, or the JSearch publisher) |
| `title` / `company` / `location` | TEXT | |
| `date_posted` | TEXT | May be null depending on source |
| `job_type` / `job_function` / `job_level` | TEXT | Not populated by every source |
| `is_remote` | INTEGER | `1` = remote, `0` = on-site |
| `min_amount` / `max_amount` / `currency` | REAL/TEXT | Salary, when the source provides it |
| `job_url` / `job_url_direct` | TEXT | Listing link / direct application link |
| `description` | TEXT | Full job description |
| `company_industry` / `company_url` | TEXT | Not populated by every source |
| `status` | TEXT | `new` / `shortlisted` / `saved` / `applied` / `rejected` |
| `score` / `score_reason` / `score_breakdown` | INTEGER/TEXT | Set by the LLM scorer; `score_breakdown` is a JSON blob (skills/company/remote/role) |
| `jd_extracted` | TEXT | Structured JD JSON, populated only under `SCORING_MODE=structured` |
| `structured_score` / `structured_score_reason` / `structured_score_breakdown` | INTEGER/TEXT | Structured-mode scores, kept in their own columns so they never overwrite the raw ones |
| `content_hash` | TEXT | Short hash of the normalized description, used for cross-source dedup (below) |
| `first_seen` / `last_seen` | TEXT | UTC timestamps |

### `referrals`

One row per saved referral contact/message, linked to a `jobs.id`.

### `candidate`

Single row (`id = 1`). Name, email, phone, LinkedIn, title, years of experience, professional summary.

### `resume`

One row per upload; the most recent is shown in the UI (`filename`, `content_type`, `raw_content`, `uploaded_at`).

### `search_criteria`

Multiple rows — each a saved search profile (`name`, `keywords`, `location`, `results`, `hours`, `remote_only`).

### `company_boards`

Multiple rows — each a company's ATS board to scan (`name`, `ats` — `greenhouse`/`lever`/`ashby`, `token`).

---

## Deduplication & upsert behavior

Duplicates are **prevented at write time**, not flagged after the fact. When saving a batch of jobs, each row is checked against existing jobs on **two independent signals** — either match means "same role, re-sighted from another source", so the existing row's `last_seen` is bumped (and `job_url_direct` backfilled if it was empty) and no new row is created:

1. **`content_hash`** (checked first) — a sha256 of the normalized job description, truncated to 12 hex characters. This catches the same posting mirrored across sources even when the title or company string differs (casing, suffixes like `(Platform)`, a recruiter's rewording of the employer name). It is skipped for descriptions under 200 characters: the hash carries no company/title scoping of its own, so a near-empty or boilerplate description could otherwise collide across genuinely different postings. Real job descriptions are far longer than that floor, so in practice this only affects rows that failed to fetch a description — those fall through to signal 2.
2. **`(title, company)`**, lowercased and normalized — the original signal, which catches the same role scraped from LinkedIn and mirrored on the company's own Greenhouse board under an identical title.

Both rules apply against the database *and* within a single incoming batch, so scanning multiple sources at once can't create duplicates either. Rows saved before `content_hash` existed have it as `NULL` until they are re-seen or backfilled with `backfill_content_hash.py`.

Re-scanning a job that already exists (same id) only updates `last_seen` and the scraped fields — `status` and `first_seen` are **never overwritten**, so your tracking data is safe across runs. A re-scrape that comes back with a blank field (e.g. a transient fetch failure) doesn't clobber a previously-populated value.

---

## Known Limitations

- **Rate limiting / blocking:** LinkedIn and Naukri may throttle or block repeated rapid scans, especially via the authenticated Playwright sources. Space out scans if you run many in sequence.
- **Selector fragility:** the authenticated LinkedIn/Naukri scrapers depend on each site's current, often-obfuscated CSS class names. A site redesign can break scraping until selectors are updated. `scanner/linkedin_playwright.py` is by far the most maintenance-heavy file in the project for this reason, and it has no automated coverage — validate changes to it with a live `debug_linkedin_scan.py` run, not the test suite.
- **Unsupported countries:** the underlying jobspy library does not recognise all countries (e.g. Kyrgyzstan) for its LinkedIn source. The scraper monkey-patches this to fall back gracefully rather than crashing.
- **Salary data:** most postings across all sources do not include salary information; `min_amount`/`max_amount` will be null for the majority of results.
- **Scoring/LLM features require an API key:** `CLAUDE_API_KEY` or `GEMINI_API_KEY` must be set for scoring, resume summary generation, and referral message drafting — the rest of the app works without one.
