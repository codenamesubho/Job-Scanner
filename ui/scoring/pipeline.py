"""The scoring run itself, shared by the Score button and post-scan auto-scoring.

Both callers previously carried their own ~90-line `_worker` closure. The two
were ~85% identical — the same SCORING_MODE branch, the same job-list assembly,
the same "iterate results, persist, count" loop — and differed only in:

  * which jobs they consider (auto-scoring looks at jobs with no score yet; the
    Score button re-scores everything with a description),
  * whether the run is cancellable,
  * the wording of the "nothing to do" message.

Those three differences are now parameters. Everything else lives here once, so
a fix to the structured/raw branching can no longer land in one copy only.
"""
from dataclasses import dataclass

from scanner import (
    extract_missing_job_requirements, get_jobs, load_resume_profile,
    parse_jd_extracted, scoreable_jobs, score_jobs, score_jobs_structured,
    scoring_mode, update_scores, update_structured_scores,
)

from ..background import BackgroundJob


@dataclass(frozen=True)
class ScoringPlan:
    """How one caller wants a scoring run to differ from the other."""

    #: True to score only jobs that have no score yet (post-scan auto-scoring);
    #: False to re-score every job that has a description (the Score button).
    only_unscored: bool
    #: Message shown when there turns out to be nothing to score.
    nothing_to_do: str
    #: Whether the run honours the job's cancel event.
    cancellable: bool = False


def _structured_jobs(scoreable) -> list[dict]:
    """Build the payload score_jobs_structured() takes, skipping rows whose
    stored jd_extracted can't be parsed back into requirements."""
    jobs_list = []
    for _, row in scoreable.iterrows():
        requirements = parse_jd_extracted(row.get("jd_extracted"))
        if requirements is not None:
            jobs_list.append({
                "id": row["id"], "requirements": requirements,
                "is_remote": bool(row.get("is_remote")),
                "title": row.get("title", ""), "company": row.get("company", ""),
            })
    return jobs_list


def _prepare_structured(job: BackgroundJob, candidate: dict, plan: ScoringPlan):
    """Extract any missing JD/resume JSON, then assemble the structured batch.

    Returns (iterator, persist_fn) or None when there is nothing to score — in
    which case the reason has already been recorded on the job.
    """
    cancel = job.cancel_event if plan.cancellable else None

    job.log("Extracting structured JD data for jobs missing it…")
    extract_missing_job_requirements(log_fn=job.log, cancel_event=cancel)
    if plan.cancellable and job.cancelled:
        return None

    resume_profile = load_resume_profile(candidate, log_fn=job.log)
    if resume_profile is None:
        job.set(skip="Structured scoring skipped — could not load a structured resume profile.")
        return None

    job.log("Checking for jobs missing a structured score…")
    pending = get_jobs(missing_structured_score=True)
    if "jd_extracted" not in pending.columns or pending.empty:
        job.set(skip="No jobs need structured scoring.")
        return None

    scoreable = scoreable_jobs(pending)
    if scoreable.empty:
        job.set(skip="No jobs need structured scoring.")
        return None

    jobs_list = _structured_jobs(scoreable)
    if not jobs_list:
        job.set(skip="No jobs have structured JD data to score against yet.")
        return None

    job.set(total=len(jobs_list))
    return (
        score_jobs_structured(resume_profile, jobs_list, log_fn=job.log, cancel_event=cancel),
        update_structured_scores,
    )


def _prepare_raw(job: BackgroundJob, candidate: dict, plan: ScoringPlan):
    """Assemble the raw-text batch. Same contract as _prepare_structured()."""
    cancel = job.cancel_event if plan.cancellable else None

    jobs = get_jobs(unscored_only=True) if plan.only_unscored else get_jobs()
    if "description" not in jobs.columns or jobs.empty:
        job.set(skip=plan.nothing_to_do)
        return None

    scoreable = scoreable_jobs(jobs)
    if scoreable.empty:
        job.set(skip=plan.nothing_to_do)
        return None

    jobs_list = scoreable[["id", "title", "company", "description", "is_remote"]].to_dict("records")
    job.set(total=len(jobs_list))
    return (
        score_jobs(candidate["summary"], jobs_list, log_fn=job.log, cancel_event=cancel),
        update_scores,
    )


def run_scoring(job: BackgroundJob, candidate: dict, plan: ScoringPlan) -> None:
    """Score jobs into `job`, persisting each batch as it completes.

    Written to be the `target` of BackgroundJob.start(), so it runs on a daemon
    thread and reports only through `job`. Results are persisted per batch
    rather than at the end, so a cancelled or failed run keeps whatever it had
    already scored.
    """
    prepare = _prepare_structured if scoring_mode() == "structured" else _prepare_raw
    prepared = prepare(job, candidate, plan)
    if prepared is None:
        return

    score_iter, persist = prepared
    for result in score_iter:
        if result:
            persist(result)
            job.add_done(len(result))
