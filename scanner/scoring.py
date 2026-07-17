"""Shared "score every unscored job" routine used by cron_scan.py and
backfill_descriptions.py (previously duplicated near line-for-line in both).

Also owns the structured-scoring pipeline (SCORING_MODE=structured):
extracting structured JD/resume JSON via a cheap model before scoring
against it with score_jobs_structured() instead of score_jobs().
"""

import json
from typing import Optional

from .database import (
    get_jobs, update_scores, update_structured_scores, scoreable_jobs,
    update_job_fields, parse_jd_extracted,
)
from .llm import (
    score_jobs, score_jobs_structured, scoring_breaker_status, scoring_mode,
    extract_job_requirements, extract_resume_profile, ResumeProfile,
)
from .profile import get_candidate, get_latest_resume, extract_text, save_candidate


def extract_missing_job_requirements(log_fn=print, limit: int | None = None,
                                      force: bool = False) -> int:
    """No-ops (returns 0) unless SCORING_MODE=structured. Populates
    jobs.jd_extracted for every scoreable job missing it — or, when
    `force=True`, re-extracts every scoreable job regardless of whether
    jd_extracted is already populated (e.g. to backfill a newly-added field
    onto jobs extracted before it existed). Either way, only the top
    `limit` of them when given. get_jobs() already orders rows by
    `COALESCE(score, -1) DESC, first_seen DESC`, so slicing the first
    `limit` rows after filtering is exactly "top N jobs by score" (jobs with
    no score yet sort after every scored one, newest-first among
    themselves). Returns the number of jobs successfully extracted."""
    if scoring_mode() != "structured":
        return 0

    scoreable = scoreable_jobs(get_jobs())
    if scoreable.empty or "jd_extracted" not in scoreable.columns:
        return 0
    if force:
        targets = scoreable
    else:
        targets = scoreable[scoreable["jd_extracted"].isna() | (scoreable["jd_extracted"] == "")]
    if limit is not None:
        targets = targets.head(limit)
    if targets.empty:
        return 0

    total = len(targets)
    verb = "Re-extracting" if force else "Extracting"
    log_fn(f"{verb} structured JD data for {total} job(s)…")
    extracted = 0
    for i, (_, row) in enumerate(targets.iterrows(), start=1):
        log_fn(f"[{i}/{total}] {verb} job {row['id']} ({row.get('title', '')})…")
        try:
            result = extract_job_requirements(row["description"], row.get("company"))
            update_job_fields(row["id"], {"jd_extracted": json.dumps(result.model_dump())})
            extracted += 1
        except Exception as e:
            log_fn(f"[{i}/{total}] Extraction failed for job {row['id']} ({row.get('title', '')}): {e}")
    log_fn(f"Extracted {extracted}/{total} job(s).")
    return extracted


def load_resume_profile(candidate: dict, log_fn) -> ResumeProfile | None:
    """Return the cached structured resume profile, extracting+caching it
    just-in-time from the latest resume file if missing."""
    raw = candidate.get("resume_extracted")
    if raw:
        try:
            return ResumeProfile(**json.loads(raw))
        except Exception:
            pass  # fall through to re-extraction

    log_fn("Structured resume profile missing — extracting now…")
    resume = get_latest_resume()
    if resume is None:
        log_fn("Structured scoring skipped — no resume on file to extract from.")
        return None
    text = extract_text(resume["filename"], resume["raw_content"])
    if not text:
        log_fn("Structured scoring skipped — could not extract resume text.")
        return None
    try:
        profile = extract_resume_profile(text)
        save_candidate(
            candidate.get("name", ""), candidate.get("email", ""),
            candidate.get("phone", ""), candidate.get("linkedin", ""),
            candidate.get("title", ""), int(candidate.get("years_exp") or 0),
            candidate.get("summary", ""),
            resume_extracted=json.dumps(profile.model_dump()),
        )
        return profile
    except Exception as e:
        log_fn(f"Structured scoring skipped — resume extraction failed: {e}")
        return None


def score_unscored_jobs(log_fn=print, limit:Optional[int]=None) -> int:
    """Score jobs that need it. Under SCORING_MODE=raw (default): every job
    with no score yet and a usable description, via the existing free-text
    score_jobs(). Under SCORING_MODE=structured: every job missing a
    structured_score (independent of whether it already has a raw score —
    see database.get_jobs' missing_structured_score param — so the two can
    be compared side by side), via score_jobs_structured(), after first
    extracting any missing structured JD/resume JSON.
    """
    candidate = get_candidate()
    if not candidate.get("summary"):
        log_fn("Scoring skipped — no candidate summary saved (add one in the Profile tab).")
        return 0

    breaker = scoring_breaker_status()
    if breaker["open"]:
        log_fn(f"Scoring skipped — model in cooldown for ~{breaker['retry_in_s']}s ({breaker['reason']}).")
        return 0

    extract_missing_job_requirements(log_fn=log_fn, limit=limit)

    if scoring_mode() == "structured":
        resume_profile = load_resume_profile(candidate, log_fn)
        if resume_profile is None:
            return 0

        pending = get_jobs(missing_structured_score=False)
        if "jd_extracted" not in pending.columns or pending.empty:
            log_fn("No jobs need structured scoring.")
            return 0
        scoreable_pending = scoreable_jobs(pending)
        if scoreable_pending.empty:
            log_fn("No jobs need structured scoring.")
            return 0
        if limit is not None:
            scoreable_pending = scoreable_pending.head(limit)

        jobs_list = []
        for _, row in scoreable_pending.iterrows():
            requirements = parse_jd_extracted(row.get("jd_extracted"))
            if requirements is not None:
                jobs_list.append({
                    "id": row["id"], "requirements": requirements,
                    "is_remote": bool(row.get("is_remote")),
                    "title": row.get("title", ""), "company": row.get("company", ""),
                })
        if not jobs_list:
            log_fn("No jobs have structured JD data to score against yet.")
            return 0

        log_fn(f"Structured-scoring {len(jobs_list)} job(s)…")
        score_iter = score_jobs_structured(resume_profile, jobs_list, log_fn=log_fn)
        update_fn = update_structured_scores
    else:
        unscored = get_jobs(unscored_only=True)
        if "description" not in unscored.columns or unscored.empty:
            log_fn("Scoring skipped — no new jobs need scoring.")
            return 0

        scoreable = scoreable_jobs(unscored)
        no_desc_count = len(unscored) - len(scoreable)
        if scoreable.empty:
            log_fn(f"Scoring skipped — no new jobs need scoring ({no_desc_count} have no description).")
            return 0
        if limit is not None:
            scoreable = scoreable.head(limit)

        jobs_list = scoreable[["id", "title", "company", "description", "is_remote"]].to_dict("records")
        log_fn(f"Scoring {len(jobs_list)} new job(s)" +
               (f" ({no_desc_count} skipped — no description)…" if no_desc_count else "…"))
        score_iter = score_jobs(candidate["summary"], jobs_list, log_fn=log_fn)
        update_fn = update_scores

    scored = 0
    for result in score_iter:
        if result:
            update_fn(result)
            scored += len(result)
    log_fn(f"Scored {scored} job(s).")
    return scored
