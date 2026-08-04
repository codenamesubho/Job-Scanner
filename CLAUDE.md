# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

All commands assume the virtualenv is active (`source venv/bin/activate`). The venv lives at `venv/` and is git-ignored.

```bash
# Install / sync dependencies
pip install -r requirements.txt

# Run the Streamlit UI
streamlit run app.py                        # opens http://localhost:8501

# Run a CLI scan (print only)
python main.py --keywords "backend engineer" --location "USA"

# CLI scan → save to database
python main.py --keywords "python developer" --location "Remote" --db

# CLI scan → save to database + CSV
python main.py --keywords "data engineer" --location "NYC" --db --save

# Daily scan across every enabled source + score new jobs (designed for cron)
python cron_scan.py

# Recover missing LinkedIn descriptions, optionally scoring newly-recovered jobs
python backfill_descriptions.py --score

# Backfill structured JD extraction for existing jobs (SCORING_MODE=structured only)
python backfill_jd_extraction.py

# Same, but only the top 50 jobs by score (useful for a large backlog)
python backfill_jd_extraction.py --top 50

# Re-extract every job, even ones already extracted (e.g. after adding a new JobRequirements field)
python backfill_jd_extraction.py --force

# Score every job that needs it (raw or structured, depending on SCORING_MODE)
python backfill_score_jobs.py

# Same, but only the top 50 jobs by score
python backfill_score_jobs.py --top 50

# Backfill jobs.content_hash for rows saved before that column existed
python backfill_content_hash.py            # missing hashes only
python backfill_content_hash.py --force    # recompute every row

# Clear the jobs and/or referrals tables
python clear_db.py --jobs

# Lint
ruff check .
```

## Environment / .env

`.env` **always exists on this machine** — it may sit outside the sandbox's readable
scope, so a tool call can report it missing when it is not. Never conclude the
project is unconfigured, and never "fix" code on the assumption that a key is
absent, based on not being able to read it.

Note where it is read from: both `scanner/config.py:4` and `scanner/llm/__init__.py:74`
call `load_dotenv(os.getenv("ENV_FILE", "~/.env"), override=True)` — that is
`~/.env` in the **home directory** by default, not a `.env` in the repo, and
`override=True` means it beats variables already exported in the shell. So
`env -u SOME_KEY` does not unset anything for this app; point `ENV_FILE` at an
empty file if you genuinely need to test the unconfigured path.
Always add new work on a branch from main and create a PR against main.
Use feature branches instead of worktree for development
Dont run git commit by yourself ever, always ask user before you commit

## Testing

```bash
pytest
```

24 test files, ~265 tests, all offline — no network, browser, or running Streamlit server.

Shared fixtures live in `tests/conftest.py` (`isolated_db` repoints `database.DB_PATH` at a
temp file; `stub_fetch_json` stubs the ATS scrapers' one HTTP call) and `tests/fakes.py`
(`FakePage`/`FakeElement`/`FakePlaywright` stand in for Playwright). Use these rather than
hand-rolling setup.

Covered: the ATS/aggregator row mapping, the scan kernel (`scanner/search.py`), shared CLI
flags, DataFrame filters, DB migrations + `save_jobs` dedup, `BackgroundJob`, the scoring
pipeline and `CircuitBreaker`, and `linkedin_playwright`'s *selector-independent* logic
(`tests/test_linkedin_pure_logic.py` — id parsing, keyword derivation, semantic-card
assembly, search-URL construction).

**Not covered, by design:** the live Playwright flows (LinkedIn/Naukri scraping and login,
`apply.py`) and the Streamlit UI. LinkedIn's obfuscated class names can only be validated
against the live site — so changes under `scanner/linkedin_playwright/` must be checked with
`python debug_linkedin_scan.py` in **both** `--keyword-mode` values, not by `pytest` alone.
For the UI, `streamlit.testing.v1.AppTest` can execute `app.py` and surface exceptions.

## Architecture

### Data flow

```
LinkedIn/Naukri/ATS boards/JSearch ──► scanner/*.py ──► pandas DataFrame
                                                              │
                                                    scanner/database.py
                                                              │
                                                       data/jobs.db (SQLite)
                                                              │
                                                        app.py (Streamlit)
```

The `scanner/` package is the only layer that touches the database or the network. `app.py` (which is 21 lines of wiring over the `ui/` package), `main.py`, `cron_scan.py`, the three `backfill_*.py` scripts, `clear_db.py`, and `debug_linkedin_scan.py` are entry points that call into it. `cli_common.py` holds the argparse flags and timestamped `log()` those CLI scripts share.

### scanner/ modules

**Job sources** — each exposes a `fetch_jobs(...)` or `search_jobs(...)` returning a DataFrame in the shared jobs-table row shape:
- **`linkedin.py`** — wraps `jobspy.scrape_jobs()` (no login). Contains a module-level monkey-patch of `LinkedIn._get_location` that catches `ValueError` for countries not in jobspy's allowlist (e.g. Kyrgyzstan) instead of crashing. Also has `fetch_missing_description()`/`backfill_missing_descriptions()` (public-page HTTP fallback for jobs saved without a description) and `display_jobs()` (CLI print helper).
- **`linkedin_playwright/`** — authenticated LinkedIn scraper/automation via a real logged-in browser session. A package, split by concern:
  - `selectors.py` — **every CSS selector and tuning constant.** This is the fragile part: LinkedIn's class names are obfuscated and change often, so a redesign should mean editing this one file.
  - `session.py` — browser/session lifecycle (`login()`, `SESSION_FILE`, `_launch`, `_save_session`) and the logging sink. `_log()` prints to stdout by default; call `set_log_fn()` before a scrape/login/referral-search to route lines into a UI log panel instead (`ui/scan_runners.py` does this in `_run_linkedin_login`). `_log_fn` is deliberately still module-level, not per-session — nothing races it, since `handle_scan_all` runs LinkedIn login alone specifically to avoid concurrent LinkedIn traffic.
  - `jobs.py` — search: scrolling, card extraction (classic *and* semantic UIs), pagination, `build_search_url()`, `search_jobs()`.
  - `descriptions.py` — three escalating strategies: side-panel click, semantic-UI click, then full navigation. Clicks are ~7x faster than navigating, so navigation is last resort.
  - `people.py` — profile scraping + `find_referral_contacts()`.
  - `messaging.py` — `send_linkedin_message()`.

  Session cookies persist to `data/playwright_sessions/linkedin.json`. Exposed through `scanner/__init__.py`'s `linkedin_login`/`linkedin_playwright_search`/`find_referral_contacts`/`send_linkedin_message` wrappers, never as the package's default `search_jobs`. **Validate any change here with `debug_linkedin_scan.py` against live LinkedIn** — the tests cover only selector-independent logic.
- **`naukri_playwright.py`** — same pattern as the LinkedIn scraper but for Naukri.com, and small enough to stay one module. Session persists to `data/playwright_sessions/naukri.json`.
- **`greenhouse.py`, `ashby.py`, `lever.py`** — free, unauthenticated public ATS job-board APIs (no login, no Playwright). Share a common shape via `scanner/ats_common.py` (`fetch_json()`, `html_to_text()`, `build_job_row()`) — each module supplies only its own field-mapping and endpoint. Registered together as `scanner.ATS_FETCHERS` (keyed by `"greenhouse"`/`"lever"`/`"ashby"`, matching the `ats` column in `company_boards`).
- **`jsearch.py`** — RapidAPI aggregator (`JSEARCH_API_KEY`) covering LinkedIn/Indeed/Glassdoor/ZipRecruiter through one keyed HTTP endpoint. Reads its API key directly from the environment at call time (not routed through `config.py`, which is CLI-only).

**Scan kernel:**
- **`search.py`** — the one implementation of "search once per keyword, tolerate a failing keyword, concatenate, save, count what's new". `SearchCriteria` (frozen dataclass; `keyword_list()` does the comma splitting), `ScanResult` (`found`/`new`, addable), `run_keyword_scan()`, `run_company_board_scan()`, and `prefixed_logger()`. **Both `cron_scan.py` and `ui/scan_runners.py` call this** — it previously existed twice, differing only in log formatting. Streamlit-free by construction, and it must never import the `scanner` facade (the facade imports it).
- **`ats_registry.py`** — `ATS_FETCHERS`, keyed by the `ats` column in `company_boards`. It lives here rather than in `scanner/__init__.py` so `search.py` can use it without a circular import.
- **`manual.py`** — `add_job_by_url()`: paste a Greenhouse/Lever/Ashby/LinkedIn job URL and save that single job (surfaced in the Jobs tab).

**Automation:**
- **`apply.py`** — "open + prefill application form" automation: launches a visible browser, navigates to a job's application URL, and best-effort fills common form fields (name, email, phone, LinkedIn, resume) from the saved candidate profile via keyword matching, falling back to an LLM match (`llm.match_form_fields`) when the heuristic finds nothing. Never clicks any submit-type control — the browser is always left open for manual review/submission. Reuses the logged-in LinkedIn session from `linkedin_playwright.SESSION_FILE`.
- **`browser.py`** — shared Playwright launch configuration (user agent, stealth init script, launch args, `launch_stealth_browser()`) plus **`SessionBrowser`**, which binds that config to one site's session file and owns `launch()`/`save()`/`has_session()`. `linkedin_playwright/session.py` and `naukri_playwright.py` each hold a `SessionBrowser`; `apply.py` calls `launch_stealth_browser()` directly. `debug_headful()` resolves `SCAN_DEBUG_HEADFUL=1` in `.env`, which makes scan browsers launch visibly.

**Scoring/LLM:**
- **`llm/`** — all LLM calls (Claude or Gemini via litellm). A package:
  - `__init__.py` — provider selection, model resolution, `execute_with_breaker()`, `parse_score_breakdown()`, and the public facade. Re-exports must stay at the bottom.
  - `_breaker.py` — **`CircuitBreaker`** (instance: `BREAKER`), per-provider rate-limit cooldown; surfaced via `scoring_breaker_status()`/`provider_breaker_status()`, and drives the Claude → Gemini fallback.
  - `_batch.py` — `run_batches()`, the concurrency/cancel/fatal-error driver **shared by both scoring modes**. Fix batching behaviour here, not in the two scorers.
  - `_tracing.py` — optional Langfuse tracing (`observe`, litellm callbacks). Separate so `observe` exists before submodules import it as a decorator.
  - `raw_scoring.py` (`score_jobs`), `structured_scoring.py` (`score_jobs_structured`), `extraction.py` (`generate_summary`, `extract_job_requirements`, `extract_resume_profile`, and the `JobRequirements`/`ResumeProfile` models), `referral.py` (`draft_referral_message`, `match_form_fields`).
  - `prompts/*.json` + `_prompt_loader.py` — prompt text lives outside the Python.
- **`scoring.py`** — `score_unscored_jobs()`: scores every DB job that has no score yet and a usable description (mode-dependent — see Structured JD/resume scoring below). Shared by `cron_scan.py` and `backfill_descriptions.py` (previously duplicated near line-for-line in both). Also owns `extract_missing_job_requirements()` and `load_resume_profile()`, the structured-mode extraction helpers.

**Persistence:**
- **`database.py`** — all `jobs` and `referrals` table logic: upsert (never overwrites `status` or `first_seen`), duplicate prevention at write time (see Deduplication below), `get_jobs()`, `scoreable_jobs()` (rows with a usable description), `update_status()`, `get_stats()`, `update_scores()`. `DB_PATH` is a module-level `Path` — tests/scripts can rebind it to a temp file.
- **`profile.py`** — four single-row-or-append tables in the same `data/jobs.db`: `candidate` (id=1 always), `search_criteria` (id=1+ for saved profiles, read by `ui/sidebar.py`'s `seed_sidebar_defaults()` before the sidebar widgets are created), `resume` (append-only, latest row wins), `company_boards` (name/ats/token rows scanned by the "Company Boards" source). Also contains `extract_text()` for best-effort PDF/DOCX parsing.
- **`filters.py`** — pure DataFrame helpers: `filter_by_keywords`, `filter_by_exclude`, `filter_by_remote_flag` (filters on the source's own `is_remote` flag, not location text).
- **`config.py`** — reads `.env` via `python-dotenv`; used only by `main.py` for CLI defaults. The UI and `cron_scan.py` read defaults from the DB (`profile.get_criteria()`) instead.

### Structured JD/resume scoring

`SCORING_MODE` (`.env`, read via `llm.scoring_mode()`) toggles between two scoring paths, kept fully independent so they can be compared side by side:

- **`raw` (default)** — the original path: `score_jobs()` sends the candidate's free-text summary and each job's raw `description` straight to the LLM.
- **`structured`** — before scoring, both sides are extracted into typed JSON via instructor/Pydantic models (`llm.JobRequirements`, `llm.ResumeProfile`), and the LLM scores JSON against JSON instead of re-reading raw text every time:
  - **`extract_missing_job_requirements()`** (`scanner/scoring.py`) — no-ops under `SCORING_MODE=raw`. Under `structured`, finds every scoreable job with an empty `jobs.jd_extracted` column, runs `llm.extract_job_requirements(description, company)` (a cheap/fast model, its own `CLAUDE_EXTRACT_MODEL`/`GEMINI_EXTRACT_MODEL` env vars — distinct from `CLAUDE_MODEL`/`GEMINI_MODEL`), and stores the resulting `JobRequirements` as JSON in `jd_extracted`. `company` is the job's company name (not JD text) — passed through so the model can use its own general knowledge of named companies to fill `company_type`/`company_size` when the JD text itself doesn't state them, the same way `score_jobs()`'s raw path recognizes company names directly. The raw description is the only place the model class distinction matters — once extracted, later scoring calls never re-send raw JD text. Takes an optional `limit` param to cap the run at the top N jobs by score — free, since `database.get_jobs()` already returns rows ordered by `COALESCE(structured_score, score, -1) DESC, first_seen DESC`, so "top N by score" is just `.head(limit)` after filtering to jobs missing `jd_extracted`. Takes an optional `force` param to re-extract every scoreable job regardless of whether `jd_extracted` is already populated — needed to backfill a newly-added `JobRequirements` field onto jobs extracted before it existed (`update_job_fields()` overwrites unconditionally, so a forced re-extraction cleanly replaces the old JSON). Logs a `[i/total]` line per job as it processes them.
  - The candidate's resume gets the same treatment once, cached as `candidate.resume_extracted` (`llm.extract_resume_profile()` → `ResumeProfile`), lazily (re-)extracted by `scoring.load_resume_profile()` when missing.
  - **`score_jobs_structured()`** (`scanner/llm/structured_scoring.py`) — scores `ResumeProfile` against a batch of `JobRequirements` using its own model class (`CLAUDE_STRUCTURED_SCORE_MODEL`/`GEMINI_STRUCTURED_SCORE_MODEL`, typically Sonnet-class vs. the Haiku-class extraction model), with the same batching/concurrency/cancel/heartbeat/circuit-breaker machinery as `score_jobs()`. Each job dict also carries `is_remote` (the DB's own boolean flag) alongside `requirements`, used as ground truth for the remote sub-score instead of the LLM-extracted `remote_policy` text, which can undersell an actually-remote role with vague wording. Results land in separate `structured_score`/`structured_score_reason`/`structured_score_breakdown` columns (via `database.update_structured_scores()`) rather than overwriting `score`/`score_reason`/`score_breakdown`, so raw and structured scores never clobber each other. Neither this nor `score_jobs()` ever asks the LLM to echo back a job's real database id — some sources (JSearch) use 400+ char opaque ids that an LLM can garble mid-string, silently failing exact-match validation. Both build a short, hash-derived per-job label instead (`llm._batch_short_id`, ephemeral — scoped to one batch call, never persisted) and resolve it back to the real id internally once results come back.
  - `database.get_jobs(missing_structured_score=True)` and `scoring_mode()`-branching inside `score_unscored_jobs()` route which path runs; `database.parse_jd_extracted()` turns a stored `jd_extracted` string back into a dict for scoring.
  - **`backfill_jd_extraction.py`** (top-level script) — backfills `jd_extracted` for jobs saved before this feature existed (or before `SCORING_MODE` was switched to `structured`); a no-op if `SCORING_MODE` isn't `structured`. See Top-level scripts below.

### Top-level scripts

- **`main.py`** — CLI entry point: search → filter → display, optionally `--db` (save) and/or `--save` (CSV export).
- **`cron_scan.py`** — headless daily job (designed for cron): runs every enabled source (jobspy, LinkedIn login, Naukri, company ATS boards, JSearch — login-based sources only run if a session already exists, since a cron job can't complete an interactive login), then calls `scanner.score_unscored_jobs()`. It has **no scan loop of its own**: it delegates to `scanner/search.py`'s `run_keyword_scan()`/`run_company_board_scan()`, tagging log lines via `prefixed_logger()`. Does not import `app.py`.
- **`backfill_descriptions.py`** — re-fetches missing LinkedIn descriptions for previously-saved jobs (`linkedin.backfill_missing_descriptions`), optionally scoring newly-recovered ones via `scanner.score_unscored_jobs()`.
- **`backfill_jd_extraction.py`** — backfills `jobs.jd_extracted` (structured JD JSON) for jobs saved before `SCORING_MODE=structured` was enabled, via `scanner.extract_missing_job_requirements()`. Extraction only — does not score; run `backfill_score_jobs.py` afterward for that. No-ops unless `SCORING_MODE=structured` is set. `--top N` caps the run to the top N jobs by score instead of every job missing extraction. `--force` re-extracts every scoreable job, including ones that already have `jd_extracted` — needed after adding a new field to `JobRequirements` so existing jobs pick it up too. See Structured JD/resume scoring above.
- **`backfill_score_jobs.py`** — scores every job that needs it via `scanner.score_unscored_jobs()` (raw or structured, depending on `SCORING_MODE`), without touching JD extraction or description backfill. Split out from `backfill_jd_extraction.py` so extraction and scoring can be run/scheduled independently. `--top N` caps the run to the top N jobs by score.
- **`backfill_content_hash.py`** — populates `jobs.content_hash` (see Deduplication below) for jobs saved before that column existed; new jobs get one automatically at save time. `--force` recomputes every row's hash, not just rows missing one.
- **`clear_db.py`** — deletes rows from `jobs` and/or `referrals`, using the same `scanner.database.DB_PATH` as everything else.

### Database

Single file `data/jobs.db`. All tables are created lazily on first connect. `database.connect()` is the public entry point (with `_connect` kept as an internal alias); `profile._connect()` calls it and then creates its own tables on top — both modules share the same file path via `DB_PATH`, which is a module-level `Path` that tests rebind (see the `isolated_db` fixture).

### Deduplication

Duplicates are prevented at write time, not flagged after the fact. `database.save_jobs()` checks TWO independent signals — via the `_CanonicalIndex` helper, which holds the three lookups (known ids, the normalized `(title, company)` key, the `content_hash`) that must be updated together for within-batch dedup to work — for an existing row under a *different* id — either match treats the incoming row as a re-sighting: the existing row's `last_seen` is bumped and its `job_url_direct` is backfilled if it didn't have one yet, and no new row is inserted.

1. **`content_hash`** (checked first) — sha256 of the normalized job description, truncated to 12 hex chars (`database.content_hash()`), stored per row and compared against existing rows' hashes. Catches the same posting mirrored across sources even when title/company differ (casing, suffixes like "(Platform)"). Returns `None` — and is skipped — for descriptions under `_CONTENT_HASH_MIN_CHARS` (200 chars): the hash has no company/title scoping of its own, so a short/boilerplate description could otherwise collide across genuinely different postings at different companies; real JD text is always well over that floor, so this only affects near-empty descriptions, which fall through to signal 2 instead. Existing rows saved before this column existed have `content_hash IS NULL` until re-seen (upsert) or backfilled (`backfill_content_hash.py`).
2. **`(title, company)`**, lowercased and normalized — the original signal, e.g. the same role scraped from LinkedIn and mirrored on its Greenhouse board under an identical title/company.

Same-id rows upsert normally, with `status` and `first_seen` never overwritten. Both rules also apply within a single incoming batch, so combining multiple sources in one `save_jobs()` call can't create duplicates either.

### UI structure (`ui/`)

`app.py` is 21 lines: page config, `seed_sidebar_defaults()`, `render_sidebar()`, two tabs. Everything else lives in `ui/`:

- **`sidebar.py`** — search settings + scan buttons. Returns `(SearchCriteria, ScanRequest)` — two typed objects, not a positional tuple. `seed_sidebar_defaults()` must run **before** the sidebar widgets are created: writing to a widget's `session_state` key after its widget exists raises `StreamlitAPIException`.
- **`jobs_tab.py`** — stats row, "Add job by URL", filters (search / status / remote / company type), the minimum-score slider, and the job table. The table's primary score column is `structured_score`, falling back to raw `score`; the slider and sort key off the same fallback.
- **`detail_panel.py`** — the selected-job dialog: description, score display, status editor (`_render_status_editor` → `update_status()` then `st.rerun()`), Apply, Referrals.
- **`profile_tab.py`**, **`referrals.py`** — candidate/resume/search-profiles/company-boards, and referral discovery → draft → send.
- **`scoring/`** — `pipeline.py` holds the actual scoring run, shared by both callers via a `ScoringPlan` (which jobs, cancellable, nothing-to-do message); `auto_score.py` (post-scan), `score_button.py` (sidebar), `display.py` (score panel + Rescore).
- **`scan_handlers.py`** (main-thread orchestration, progress bars) vs **`scan_runners.py`** (Streamlit-free per-source setup, delegating to `scanner/search.py`). The `_run_` prefix appears in both — it does **not** indicate thread-safety; the split is which thread may touch `st.*`.
- **`background.py`** — `BackgroundJob`/`JobState`: owns the lock, the state and the worker thread together. `snapshot()` returns a copy so the main thread reads without locking; `set()` rejects unknown field names; `finished` is set in a `finally` so a crashed worker can't leave the UI polling forever. **Use this for any new background work** — it replaced three hand-rolled lock+dict copies.
- **`models.py`** (`ScanRequest`), **`session_keys.py`** (names for every `st.session_state` entry — use these, not string literals), **`constants.py`** (thresholds, poll intervals, log-box heights).

Streamlit renders all tab bodies in one script run — `st.tabs()` only hides/shows. There is no `st.stop()` anywhere; empty-state guards use `if/else`.

Background work never touches `st.*` from a worker thread (Streamlit widgets aren't thread-safe): workers report into a `BackgroundJob`, and only the main thread reads its `snapshot()` to update progress bars and log placeholders. `_render_score_button` is deliberately different — it persists its job in `st.session_state` and ticks via `sleep` + `st.rerun()` rather than a blocking loop, because a blocking loop would never return to Streamlit's event loop and the Cancel click could not be delivered until the whole run finished.
