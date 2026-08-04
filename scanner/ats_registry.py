"""Registry of company-ATS-board scrapers, keyed by the `ats` value stored in the
company_boards table (see profile.save_company_board).

This lives in its own module rather than in `scanner/__init__.py` so that
`scanner.search` can import it without a circular import — the package facade
imports `search`, so `search` must never import the facade back.
"""
from .ashby import fetch_jobs as ashby_fetch_jobs
from .greenhouse import fetch_jobs as greenhouse_fetch_jobs
from .lever import fetch_jobs as lever_fetch_jobs

ATS_FETCHERS = {
    "greenhouse": greenhouse_fetch_jobs,
    "lever":      lever_fetch_jobs,
    "ashby":      ashby_fetch_jobs,
}
