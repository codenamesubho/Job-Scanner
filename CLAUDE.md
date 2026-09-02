# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

All commands assume the virtualenv is active (`source venv/bin/activate`). The venv lives at `venv/` and is git-ignored.

```bash
# Install / sync dependencies
pip install -r requirements.txt

# One-time: enable the "Apply" button (drives the separate Autofill-Job-Application
# project — see scanner/autofill_bridge.py). Not in requirements.txt; not on PyPI.
pip install -e /path/to/Autofill-Job-Application

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

# Clear the jobs and/or referrals tables
python clear_db.py --jobs
```
Always add new work on a branch from main and create a PR against main.
Do not create a git worktree for this. Work directly in the current checkout — `git checkout -b <branch>` there — rather than in a separate worktree directory.
Dont run git commit by yourself ever, always ask user before you commit

## Testing

```bash
pytest
```

The suite (`tests/`) covers the pure-logic parts of the ATS/aggregator scrapers (`greenhouse.py`, `ashby.py`, `lever.py`, `jsearch.py`), the shared Playwright launch helper (`scanner/browser.py`), and the row-parsing helpers in `naukri_playwright.py` — all via mocked HTTP/fake page objects, no network or browser required. There is no coverage for the live Playwright scraping/login flows (`linkedin_playwright.py`, `naukri_playwright.py`'s `search_jobs`/`login`, `apply.py`) or for `app.py` itself — those need a real session/browser or a running Streamlit server to exercise.

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

The `scanner/` package is the only layer that touches the database or the network. `app.py`, `main.py`, `cron_scan.py`, `backfill_descriptions.py`, and `clear_db.py` are entry points that call into it.

### scanner/ modules

**Job sources** — each exposes a `fetch_jobs(...)` or `search_jobs(...)` returning a DataFrame in the shared jobs-table row shape:
- **`linkedin.py`** — wraps `jobspy.scrape_jobs()` (no login). Contains a module-level monkey-patch of `LinkedIn._get_location` that catches `ValueError` for countries not in jobspy's allowlist (e.g. Kyrgyzstan) instead of crashing. Also has `fetch_missing_description()`/`backfill_missing_descriptions()` (public-page HTTP fallback for jobs saved without a description) and `display_jobs()` (CLI print helper).
- **`linkedin_playwright.py`** — authenticated LinkedIn scraper/automation via a real logged-in browser session (`login()`, `search_jobs()`, `find_referral_contacts()`, `send_linkedin_message()`). Session cookies persist to `data/playwright_sessions/linkedin.json`. Exposed through `scanner/__init__.py`'s `linkedin_login`/`linkedin_playwright_search`/`find_referral_contacts`/`send_linkedin_message` wrappers, never as the package's default `search_jobs`. Progress/status lines go through its `_log()`, which prints to stdout by default — call `set_log_fn()` before a scrape/login/referral-search to route those lines into a UI log panel instead (app.py does this in `_run_linkedin_login`). By far the most maintenance-fragile file in the project: it depends on LinkedIn's unstable, obfuscated CSS class names, with extensive selector-fallback chains throughout.
- **`naukri_playwright.py`** — same pattern as `linkedin_playwright.py` but for Naukri.com. Session persists to `data/playwright_sessions/naukri.json`.
- **`greenhouse.py`, `ashby.py`, `lever.py`** — free, unauthenticated public ATS job-board APIs (no login, no Playwright). Share a common shape via `scanner/ats_common.py` (`fetch_json()`, `html_to_text()`, `build_job_row()`) — each module supplies only its own field-mapping and endpoint. Registered together as `scanner.ATS_FETCHERS` (keyed by `"greenhouse"`/`"lever"`/`"ashby"`, matching the `ats` column in `company_boards`).
- **`jsearch.py`** — RapidAPI aggregator (`JSEARCH_API_KEY`) covering LinkedIn/Indeed/Glassdoor/ZipRecruiter through one keyed HTTP endpoint. Reads its API key directly from the environment at call time (not routed through `config.py`, which is CLI-only).

**Automation:**
- **`apply.py`** — the original "open + prefill application form" automation: launches a visible browser, navigates to a job's application URL, and best-effort fills common form fields (name, email, phone, LinkedIn, resume) from the saved candidate profile via keyword matching, falling back to an LLM match (`llm.match_form_fields`) when the heuristic finds nothing. Never clicks any submit-type control — the browser is always left open for manual review/submission. Reuses the logged-in LinkedIn session from `linkedin_playwright.SESSION_FILE`. **Deprecated — no longer wired into the UI** (superseded by `autofill_bridge.py`); left in the repo unused rather than deleted.
- **`autofill_bridge.py`** — the current Apply engine, driving the separate [Autofill-Job-Application](https://github.com/codenamesubho/Autofill-Job-Application) project instead of `apply.py`'s keyword heuristic. That project's `autofill-fill` agent (LLM + `browser-use`) discovers *every* question on a real ATS form — not just the handful `apply.py` knew how to match — and answers each one from a markdown candidate-profile doc, with its own code-level guardrails (a restricted tool registry + a CDP-level submit blocker) that make it structurally incapable of clicking submit, matching `apply.py`'s "never auto-submit" rule. Not a `requirements.txt` dependency (not published to PyPI) — install it into this venv separately: `pip install -e /path/to/Autofill-Job-Application`. `autofill_bridge.run_apply(job_url, log_fn, cancel_event)` shells out to its `autofill-fill` console script (its own async `browser-use` agent loop, so subprocess rather than an in-process import — same reasoning `apply.py` uses for running Playwright's sync API on a dedicated thread), builds the required markdown candidate doc lazily at `data/autofill_context.md` from `profile.get_candidate()` (`build_context_markdown()` — only overwrites on an explicit `force=True`, since Autofill-Job-Application treats that file as hand-editable "source of truth"), and derives `AUTOFILL_LLM_*` env vars from this project's own `CLAUDE_API_KEY`/`CLAUDE_MODEL` (`resolve_llm_env()`) so nothing new needs configuring by default — mapped through `AUTOFILL_LLM_PROVIDER=litellm` + `AUTOFILL_LLM_BASE_URL=http://localhost:8317`, not `AUTOFILL_LLM_PROVIDER=anthropic`, since `CLAUDE_API_KEY` here is only valid against this project's local CLIProxyAPI bridge, not api.anthropic.com directly (see `scanner/llm/__init__.py`'s `_PROVIDER_CONFIG`). Set `AUTOFILL_LLM_API_KEY` explicitly in `.env` to override the derived config (e.g. to point Autofill at a different provider/model than the scorer uses).
- **`browser.py`** — shared Playwright launch configuration (user agent, stealth init script, launch args, and a `launch_stealth_browser()` helper) used by `apply.py`, `linkedin_playwright.py`, and `naukri_playwright.py` so the three don't each redefine the same boilerplate.

**Scoring/LLM:**
- **`llm.py`** — all LLM calls (Claude or Gemini, OpenAI-compatible API): resume summary generation, batch job scoring (`score_jobs`), referral message drafting, form-field matching for `apply.py`, and `parse_score_breakdown()` (turns a stored `score_breakdown` string — current JSON format or legacy pipe format — into a display-ready structure). Owns provider selection and a per-provider circuit breaker (`scoring_breaker_status()`/`provider_breaker_status()`) that falls back Claude → Gemini on a rate limit.
- **`scoring.py`** — `score_unscored_jobs()`: scores every DB job that has no score yet and a usable description (mode-dependent — see Structured JD/resume scoring below). Shared by `cron_scan.py` and `backfill_descriptions.py` (previously duplicated near line-for-line in both). Also owns `extract_missing_job_requirements()` and `load_resume_profile()`, the structured-mode extraction helpers.

**Persistence:**
- **`database.py`** — all `jobs` and `referrals` table logic: upsert (never overwrites `status` or `first_seen`), duplicate prevention at write time (see Deduplication below), `get_jobs()`, `scoreable_jobs()` (rows with a usable description), `update_status()`, `get_stats()`, `update_scores()`. `DB_PATH` is a module-level `Path` — tests/scripts can rebind it to a temp file.
- **`profile.py`** — four single-row-or-append tables in the same `data/jobs.db`: `candidate` (id=1 always), `search_criteria` (id=1+ for saved profiles, read by `app.py` before sidebar widgets render to set their defaults), `resume` (append-only, latest row wins), `company_boards` (name/ats/token rows scanned by the "Company Boards" source). Also contains `extract_text()` for best-effort PDF/DOCX parsing.
- **`filters.py`** — pure DataFrame helpers: `filter_by_keywords`, `filter_by_exclude`, `filter_by_remote_flag` (filters on the source's own `is_remote` flag, not location text).
- **`config.py`** — reads `.env` via `python-dotenv`; used only by `main.py` for CLI defaults. The UI and `cron_scan.py` read defaults from the DB (`profile.get_criteria()`) instead.

### Structured JD/resume scoring

`SCORING_MODE` (`.env`, read via `llm.scoring_mode()`) toggles between two scoring paths, kept fully independent so they can be compared side by side:

- **`raw` (default)** — the original path: `score_jobs()` sends the candidate's free-text summary and each job's raw `description` straight to the LLM.
- **`structured`** — before scoring, both sides are extracted into typed JSON via instructor/Pydantic models (`llm.JobRequirements`, `llm.ResumeProfile`), and the LLM scores JSON against JSON instead of re-reading raw text every time:
  - **`extract_missing_job_requirements()`** (`scanner/scoring.py`) — no-ops under `SCORING_MODE=raw`. Under `structured`, finds every scoreable job with an empty `jobs.jd_extracted` column, runs `llm.extract_job_requirements(description, company)` (a cheap/fast model, its own `CLAUDE_EXTRACT_MODEL`/`GEMINI_EXTRACT_MODEL` env vars — distinct from `CLAUDE_MODEL`/`GEMINI_MODEL`), and stores the resulting `JobRequirements` as JSON in `jd_extracted`. `company` is the job's company name (not JD text) — passed through so the model can use its own general knowledge of named companies to fill `company_type`/`company_size` when the JD text itself doesn't state them, the same way `score_jobs()`'s raw path recognizes company names directly. The raw description is the only place the model class distinction matters — once extracted, later scoring calls never re-send raw JD text. Takes an optional `limit` param to cap the run at the top N jobs by score — free, since `database.get_jobs()` already returns rows ordered by `COALESCE(score, -1) DESC, first_seen DESC`, so "top N by score" is just `.head(limit)` after filtering to jobs missing `jd_extracted`. Takes an optional `force` param to re-extract every scoreable job regardless of whether `jd_extracted` is already populated — needed to backfill a newly-added `JobRequirements` field onto jobs extracted before it existed (`update_job_fields()` overwrites unconditionally, so a forced re-extraction cleanly replaces the old JSON). Logs a `[i/total]` line per job as it processes them.
  - The candidate's resume gets the same treatment once, cached as `candidate.resume_extracted` (`llm.extract_resume_profile()` → `ResumeProfile`), lazily (re-)extracted by `scoring.load_resume_profile()` when missing.
  - **`score_jobs_structured()`** (`scanner/llm.py`) — scores `ResumeProfile` against a batch of `JobRequirements` using its own model class (`CLAUDE_STRUCTURED_SCORE_MODEL`/`GEMINI_STRUCTURED_SCORE_MODEL`, typically Sonnet-class vs. the Haiku-class extraction model), with the same batching/concurrency/cancel/heartbeat/circuit-breaker machinery as `score_jobs()`. Each job dict also carries `is_remote` (the DB's own boolean flag) alongside `requirements`, used as ground truth for the remote sub-score instead of the LLM-extracted `remote_policy` text, which can undersell an actually-remote role with vague wording. Results land in separate `structured_score`/`structured_score_reason`/`structured_score_breakdown` columns (via `database.update_structured_scores()`) rather than overwriting `score`/`score_reason`/`score_breakdown`, so raw and structured scores never clobber each other. Neither this nor `score_jobs()` ever asks the LLM to echo back a job's real database id — some sources (JSearch) use 400+ char opaque ids that an LLM can garble mid-string, silently failing exact-match validation. Both build a short, hash-derived per-job label instead (`llm._batch_short_id`, ephemeral — scoped to one batch call, never persisted) and resolve it back to the real id internally once results come back.
  - `database.get_jobs(missing_structured_score=True)` and `scoring_mode()`-branching inside `score_unscored_jobs()` route which path runs; `database.parse_jd_extracted()` turns a stored `jd_extracted` string back into a dict for scoring.
  - **`backfill_jd_extraction.py`** (top-level script) — backfills `jd_extracted` for jobs saved before this feature existed (or before `SCORING_MODE` was switched to `structured`); a no-op if `SCORING_MODE` isn't `structured`. See Top-level scripts below.

### Top-level scripts

- **`main.py`** — CLI entry point: search → filter → display, optionally `--db` (save) and/or `--save` (CSV export).
- **`cron_scan.py`** — headless daily job (designed for cron): scans every enabled source (jobspy, LinkedIn login, Naukri, company ATS boards, JSearch — login-based sources only run if a session already exists, since a cron job can't complete an interactive login), saves new jobs, then calls `scanner.score_unscored_jobs()`. Does not import `app.py`.
- **`backfill_descriptions.py`** — re-fetches missing LinkedIn descriptions for previously-saved jobs (`linkedin.backfill_missing_descriptions`), optionally scoring newly-recovered ones via `scanner.score_unscored_jobs()`.
- **`backfill_jd_extraction.py`** — backfills `jobs.jd_extracted` (structured JD JSON) for jobs saved before `SCORING_MODE=structured` was enabled, via `scanner.extract_missing_job_requirements()`. Extraction only — does not score; run `backfill_score_jobs.py` afterward for that. No-ops unless `SCORING_MODE=structured` is set. `--top N` caps the run to the top N jobs by score instead of every job missing extraction. `--force` re-extracts every scoreable job, including ones that already have `jd_extracted` — needed after adding a new field to `JobRequirements` so existing jobs pick it up too. See Structured JD/resume scoring above.
- **`backfill_score_jobs.py`** — scores every job that needs it via `scanner.score_unscored_jobs()` (raw or structured, depending on `SCORING_MODE`), without touching JD extraction or description backfill. Split out from `backfill_jd_extraction.py` so extraction and scoring can be run/scheduled independently. `--top N` caps the run to the top N jobs by score.
- **`backfill_content_hash.py`** — populates `jobs.content_hash` (see Deduplication below) for jobs saved before that column existed; new jobs get one automatically at save time. `--force` recomputes every row's hash, not just rows missing one.
- **`clear_db.py`** — deletes rows from `jobs` and/or `referrals`, using the same `scanner.database.DB_PATH` as everything else.

### Database

Single file `data/jobs.db`. All tables are created lazily on first `_connect()` call. `profile._connect()` calls `database._connect()` and then creates its own tables on top — both modules share the same file path via `DB_PATH`.

`autofill_bridge.py` writes non-DB artifacts alongside it, all resolved against `database.DB_PATH`'s directory rather than a hardcoded `data/` so they follow `DB_PATH` if it's ever rebound (e.g. in tests): `data/autofill_context.md` (the candidate profile doc, hand-editable — see above), `data/tmp/` (per-run resume/URL scratch files), and `data/autofill_runs/<job-slug>/` (each Apply run's `autofill-fill` output, including the `fills/*.json` this project parses for its success/escalated/failed summary).

### Deduplication

Duplicates are prevented at write time, not flagged after the fact. `database.save_jobs()` checks TWO independent signals for an existing row under a *different* id — either match treats the incoming row as a re-sighting: the existing row's `last_seen` is bumped and its `job_url_direct` is backfilled if it didn't have one yet, and no new row is inserted.

1. **`content_hash`** (checked first) — sha256 of the normalized job description, truncated to 12 hex chars (`database.content_hash()`), stored per row and compared against existing rows' hashes. Catches the same posting mirrored across sources even when title/company differ (casing, suffixes like "(Platform)"). Returns `None` — and is skipped — for descriptions under `_CONTENT_HASH_MIN_CHARS` (200 chars): the hash has no company/title scoping of its own, so a short/boilerplate description could otherwise collide across genuinely different postings at different companies; real JD text is always well over that floor, so this only affects near-empty descriptions, which fall through to signal 2 instead. Existing rows saved before this column existed have `content_hash IS NULL` until re-seen (upsert) or backfilled (`backfill_content_hash.py`).
2. **`(title, company)`**, lowercased and normalized — the original signal, e.g. the same role scraped from LinkedIn and mirrored on its Greenhouse board under an identical title/company.

Same-id rows upsert normally, with `status` and `first_seen` never overwritten. Both rules also apply within a single incoming batch, so combining multiple sources in one `save_jobs()` call can't create duplicates either.

### UI structure (`app.py`)

Streamlit renders all tab bodies in one script run — `st.tabs()` only hides/shows. There is no `st.stop()` anywhere; empty-state guards use `if/else`. Saved `search_criteria` is read from the DB **before** any sidebar widget is defined so the widget `value=`/`key=` defaults reflect the user's saved defaults.

The jobs table uses `st.dataframe(..., selection_mode="single-row", on_select="rerun")`; a separate status `selectbox` + save button (`_render_status_editor`) calls `update_status()` directly on click, followed by `st.rerun()`.

Background work (scanning, scoring) runs in daemon threads that never touch `st.*` directly — each writes into a lock-protected shared dict, and only the main thread reads that dict to update progress bars/log placeholders in a polling loop (see `_run_parallel_sources`, `_auto_score_new`). The one exception is `_render_score_button`, which persists its job state in `st.session_state` and polls via repeated `st.rerun()` calls instead of a blocking loop — this is deliberate: a blocking loop would prevent a second button click (Cancel) from ever being processed until the whole run finished.
