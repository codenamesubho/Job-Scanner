import pandas as pd


def filter_by_keywords(jobs: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if not keywords or jobs.empty:
        return jobs
    mask = jobs["title"].str.lower().str.contains("|".join(k.lower() for k in keywords), na=False)
    return jobs[mask]


def filter_by_exclude(jobs: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    if not exclude or jobs.empty:
        return jobs
    mask = ~jobs["title"].str.lower().str.contains("|".join(e.lower() for e in exclude), na=False)
    return jobs[mask]


def filter_by_remote_flag(jobs: pd.DataFrame) -> pd.DataFrame:
    """Keep only jobs the source itself flagged as remote (the `is_remote`
    column), as opposed to guessing from the location text."""
    if jobs.empty:
        return jobs
    return jobs[jobs["is_remote"] == True]
