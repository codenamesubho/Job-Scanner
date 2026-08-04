from .linkedin import search_jobs, display_jobs, backfill_missing_descriptions
from .filters import filter_by_keywords, filter_by_exclude, filter_by_remote_flag
from .database import (
    save_jobs, get_jobs, update_status, get_stats, update_scores,
    update_structured_scores, parse_jd_extracted,
    save_referral, get_referrals, delete_referral, scoreable_jobs,
    backfill_content_hashes, reject_low_scores,
)
from .profile import (
    save_candidate, get_candidate,
    save_resume, get_latest_resume, extract_text,
    save_criteria, get_criteria, get_all_criteria, delete_criteria,
    save_company_board, get_company_boards, delete_company_board,
)
from .greenhouse import fetch_jobs as greenhouse_fetch_jobs
from .lever import fetch_jobs as lever_fetch_jobs
from .ashby import fetch_jobs as ashby_fetch_jobs
from .jsearch import search_jobs as jsearch_search_jobs
from .manual import add_job_by_url

from .llm import (
    generate_summary, score_jobs, BATCH_SIZE, draft_referral_message,
    scoring_breaker_status, parse_score_breakdown,
    scoring_mode, extract_job_requirements, extract_resume_profile,
    score_jobs_structured, JobRequirements, ResumeProfile,
)
from .scoring import score_unscored_jobs, extract_missing_job_requirements, load_resume_profile

# Registry of company-ATS-board scrapers, keyed by the `ats` value stored in
# the company_boards table (see profile.save_company_board). Shared by
# app.py's "Company Boards" scan button and cron_scan.py.
ATS_FETCHERS = {
    "greenhouse": greenhouse_fetch_jobs,
    "lever":      lever_fetch_jobs,
    "ashby":      ashby_fetch_jobs,
}


def linkedin_login(email: str, password: str) -> bool:
    from .linkedin_playwright import login
    return login(email, password)


def linkedin_playwright_search(keywords, location, results_wanted=25, hours_old=72, on_page_done=None):
    from .linkedin_playwright import search_jobs as _search
    return _search(keywords, location, results_wanted=results_wanted, hours_old=hours_old, on_page_done=on_page_done)


def find_referral_contacts(company: str, job_title: str, limit: int = 10):
    from .linkedin_playwright import find_referral_contacts as _find
    return _find(company, job_title, limit=limit)


def send_linkedin_message(profile_url: str, message: str, auto_send: bool = True) -> bool:
    from .linkedin_playwright import send_linkedin_message as _send
    return _send(profile_url, message, auto_send=auto_send)


def naukri_login(email: str, password: str) -> bool:
    from .naukri_playwright import login
    return login(email, password)


def naukri_search(keywords, location, results_wanted=25, experience=0, hours_old=None):
    # hours_old accepted (not used) for call-signature compatibility with
    # app.py's _do_scan_core, which calls every scan_fn uniformly with
    # hours_old — Naukri has no time-window filter, only experience.
    from .naukri_playwright import search_jobs as _search
    return _search(keywords, location, results_wanted=results_wanted, experience=experience)


def apply_and_prefill(url: str, log_fn=None) -> dict:
    from .apply import apply_and_prefill as _apply
    return _apply(url, log_fn=log_fn)
