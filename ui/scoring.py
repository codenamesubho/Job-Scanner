import json
import threading
import time

import pandas as pd
import streamlit as st

from scanner import (
    extract_job_requirements, extract_missing_job_requirements, get_candidate,
    get_jobs, load_resume_profile, parse_jd_extracted, parse_score_breakdown,
    scoreable_jobs, score_jobs, score_jobs_structured, scoring_breaker_status,
    scoring_mode, update_scores, update_structured_scores,
)
from scanner.database import update_job_fields

from .constants import (
    AUTO_SCORE_LOG_TAIL_LINES, LOG_TAIL_LINES, POLL_INTERVAL_S,
    SCORE_BUTTON_POLL_S, SCORE_GOOD_THRESHOLD, SCORE_OK_THRESHOLD,
)


def _auto_score_new() -> None:
    """Phase 2 (dedup already-scored jobs) + Phase 3 (score what's left) of the
    scan pipeline. Phase 1 (fetch) is whatever runner called this — fetching
    itself already dedupes cross-source at insert time (scanner.database.save_jobs);
    this phase additionally skips jobs that already have a score from a
    previous scan, so only genuinely new jobs reach the LLM.

    Under SCORING_MODE=structured, extracts any missing structured JD/resume
    JSON first, then scores against structured_score instead of score (see
    scanner.scoring.score_unscored_jobs for the same two-mode logic used by
    the CLI/cron path). Extraction, resume-profile loading, and job discovery
    all run inside the same background daemon thread as the actual scoring
    (rather than blocking the Streamlit script-runner thread beforehand) so
    their progress reaches the log box and the UI stays responsive while a
    large batch of new jobs is extracted one at a time.
    """
    cand_data = get_candidate()
    if not cand_data.get("summary"):
        st.caption("⚠ No candidate summary — skipping scoring. Add one in Profile.")
        return

    breaker = scoring_breaker_status()
    if breaker["open"]:
        st.error(
            f"⏳ Scoring not available — model in cooldown for ~{breaker['retry_in_s']}s "
            f"({breaker['reason']})."
        )
        return

    structured = scoring_mode() == "structured"

    # score_jobs()/score_jobs_structured() run multiple batches concurrently,
    # each with its own heartbeat sub-thread logging "still running…" — so
    # all logging here must go through a locked shared dict, with only the
    # main thread ever touching the log_box/progress widgets (same pattern
    # as handle_scan_all).
    state_lock = threading.Lock()
    state = {"log": [], "scored": 0, "total": 0, "done": False, "error": None, "skip": None}

    def _log_fn(msg: str) -> None:
        with state_lock:
            state["log"].append(msg)

    def _worker() -> None:
        try:
            resume_profile = None
            if structured:
                _log_fn("Extracting structured JD data for jobs missing it…")
                extract_missing_job_requirements(log_fn=_log_fn)
                resume_profile = load_resume_profile(cand_data, log_fn=_log_fn)
                if resume_profile is None:
                    with state_lock:
                        state["skip"] = "Structured scoring skipped — could not load a structured resume profile."
                    return

                _log_fn("Checking for jobs missing a structured score…")
                pending = get_jobs(missing_structured_score=True)
                if "jd_extracted" not in pending.columns or pending.empty:
                    with state_lock:
                        state["skip"] = "No jobs need structured scoring."
                    return
                scoreable = scoreable_jobs(pending)
                if scoreable.empty:
                    with state_lock:
                        state["skip"] = "No jobs need structured scoring."
                    return
                jobs_list = []
                for _, row in scoreable.iterrows():
                    requirements = parse_jd_extracted(row.get("jd_extracted"))
                    if requirements is not None:
                        jobs_list.append({
                            "id": row["id"], "requirements": requirements,
                            "is_remote": bool(row.get("is_remote")),
                            "title": row.get("title", ""), "company": row.get("company", ""),
                        })
                if not jobs_list:
                    with state_lock:
                        state["skip"] = "No jobs have structured JD data to score against yet."
                    return
                with state_lock:
                    state["total"] = len(jobs_list)
                score_iter = score_jobs_structured(resume_profile, jobs_list, log_fn=_log_fn)
                update_fn = update_structured_scores
            else:
                unscored = get_jobs(unscored_only=True)
                if "description" not in unscored.columns or unscored.empty:
                    with state_lock:
                        state["skip"] = "No new jobs need scoring."
                    return
                scoreable = scoreable_jobs(unscored)
                if scoreable.empty:
                    with state_lock:
                        state["skip"] = "No new jobs need scoring."
                    return
                jobs_list = scoreable[["id", "title", "company", "description", "is_remote"]].to_dict("records")
                with state_lock:
                    state["total"] = len(jobs_list)
                score_iter = score_jobs(cand_data["summary"], jobs_list, log_fn=_log_fn)
                update_fn = update_scores

            for result in score_iter:
                if result:
                    update_fn(result)
                    with state_lock:
                        state["scored"] += len(result)
        except Exception as e:
            with state_lock:
                state["error"] = str(e)
        finally:
            with state_lock:
                state["done"] = True

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    progress = st.progress(0, text="Checking for jobs to score…")
    log_box  = st.empty()

    while thread.is_alive():
        with state_lock:
            scored = state["scored"]
            total  = state["total"]
            lines  = list(state["log"][-AUTO_SCORE_LOG_TAIL_LINES:])
        if total:
            progress.progress(min(scored / total, 1.0), text=f"Scoring {scored} / {total} job(s)…")
        log_box.code("\n".join(lines))
        time.sleep(POLL_INTERVAL_S)

    with state_lock:
        scored = state["scored"]
        error  = state["error"]
        skip   = state["skip"]

    progress.empty()
    log_box.empty()
    if error:
        st.warning(f"Scoring failed: {error}")
    elif skip:
        st.caption(skip)
    else:
        st.info(f"Scored {scored} new job(s).")


def _render_score_button(score_col=st) -> None:
    """Runs scoring in a background thread, tracked in session_state, and
    polls via short sleep + st.rerun() ticks (rather than one long blocking
    loop) so the button click that starts a re-run can actually be delivered
    and processed while a job is in progress — that's what lets a second
    click on this same button act as Cancel instead of being ignored until
    the whole run finishes.

    `score_col` is the column the button itself renders into (so it can sit
    aligned with the other filter controls); progress/log output below the
    button always renders full-width via the plain `st` module.
    """
    job = st.session_state.get("_score_job")

    if job is None:
        if not score_col.button("🎯 Score Jobs", use_container_width=True):
            return

        cand    = get_candidate()
        summary = cand.get("summary", "")
        if not summary:
            st.warning("No candidate summary found. Generate one in the Profile tab first.")
            return

        breaker = scoring_breaker_status()
        if breaker["open"]:
            st.error(
                f"⏳ Scoring not available — model in cooldown for ~{breaker['retry_in_s']}s "
                f"({breaker['reason']})."
            )
            return

        structured = scoring_mode() == "structured"

        cancel_event = threading.Event()
        state_lock   = threading.Lock()
        state = {"log": [], "scored": 0, "total": 0, "done": False, "error": None, "skip": None}

        def _log_fn(msg: str) -> None:
            with state_lock:
                state["log"].append(msg)

        def _worker() -> None:
            try:
                resume_profile = None
                if structured:
                    extract_missing_job_requirements(log_fn=_log_fn, cancel_event=cancel_event)
                    if cancel_event.is_set():
                        return
                    resume_profile = load_resume_profile(cand, log_fn=_log_fn)
                    if resume_profile is None:
                        with state_lock:
                            state["skip"] = "Structured scoring skipped — could not load a structured resume profile."
                        return
                    pending = get_jobs(missing_structured_score=True)
                    scoreable = scoreable_jobs(pending) if "jd_extracted" in pending.columns else pending.iloc[0:0]
                    if scoreable.empty:
                        with state_lock:
                            state["skip"] = "No jobs need structured scoring."
                        return
                    jobs_list = []
                    for _, row in scoreable.iterrows():
                        requirements = parse_jd_extracted(row.get("jd_extracted"))
                        if requirements is not None:
                            jobs_list.append({
                                "id": row["id"], "requirements": requirements,
                                "is_remote": bool(row.get("is_remote")),
                                "title": row.get("title", ""), "company": row.get("company", ""),
                            })
                    if not jobs_list:
                        with state_lock:
                            state["skip"] = "No jobs have structured JD data to score against yet."
                        return
                    with state_lock:
                        state["total"] = len(jobs_list)
                    score_iter = score_jobs_structured(resume_profile, jobs_list, log_fn=_log_fn, cancel_event=cancel_event)
                    update_fn = update_structured_scores
                else:
                    all_jobs  = get_jobs()
                    scoreable = scoreable_jobs(all_jobs)
                    if scoreable.empty:
                        with state_lock:
                            state["skip"] = "No jobs with descriptions to score."
                        return
                    jobs_list = scoreable[["id", "title", "company", "description", "is_remote"]].to_dict("records")
                    with state_lock:
                        state["total"] = len(jobs_list)
                    score_iter = score_jobs(summary, jobs_list, log_fn=_log_fn, cancel_event=cancel_event)
                    update_fn = update_scores

                for result in score_iter:
                    if result:
                        update_fn(result)
                        with state_lock:
                            state["scored"] += len(result)
            except Exception as e:
                with state_lock:
                    state["error"] = str(e)
            finally:
                with state_lock:
                    state["done"] = True

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        st.session_state["_score_job"] = {
            "thread": thread, "cancel_event": cancel_event, "lock": state_lock, "state": state,
        }
        st.rerun()
        return

    # A job is already running — this click cancels it instead of starting another.
    if score_col.button("🛑 Cancel Scoring", use_container_width=True):
        job["cancel_event"].set()

    with job["lock"]:
        scored = job["state"]["scored"]
        total  = job["state"]["total"]
        lines  = list(job["state"]["log"][-LOG_TAIL_LINES:])
        done   = job["state"]["done"]
        error  = job["state"]["error"]
        skip   = job["state"]["skip"]
    cancelled = job["cancel_event"].is_set()

    if total:
        st.progress(min(scored / total, 1.0), text=f"Scoring {scored} / {total} job(s)…")
    else:
        st.progress(0, text="Extracting structured JD data…" if not done else "Done.")
    st.code("\n".join(lines))

    if done:
        del st.session_state["_score_job"]
        if error:
            st.error(f"Scoring failed: {error}")
        elif skip:
            st.info(skip)
        elif cancelled:
            st.warning(f"Scoring cancelled — {scored} job(s) scored before stopping.")
        else:
            st.success(f"Scored {scored} job(s).")
    else:
        time.sleep(SCORE_BUTTON_POLL_S)
        st.rerun()


def _score_color(score: int) -> str:
    if score >= SCORE_GOOD_THRESHOLD:
        return "🟢"
    if score >= SCORE_OK_THRESHOLD:
        return "🟡"
    return "🔴"


def _render_score_display(sel: pd.Series) -> None:
    score_val = sel.get("score")
    structured_score_val = sel.get("structured_score")
    score_col, btn_col = st.columns([3, 1])
    if pd.notna(score_val):
        breakdown_raw = sel.get("score_breakdown", "") or ""
        reason        = sel.get("score_reason", "") or ""

        parsed   = parse_score_breakdown(breakdown_raw, fallback_score=score_val)
        computed = parsed["computed_score"]
        score_col.metric(label="Match Score", value=f"{_score_color(computed)} {computed} / 100")

        if parsed["items"]:
            for label, sc, mx, rsn in parsed["items"]:
                line = f"**{label}**: {sc}/{mx}"
                if rsn:
                    line += f" — {rsn}"
                st.markdown(line)
        elif parsed["legacy_lines"]:
            st.markdown("\n".join(f"- {p}" for p in parsed["legacy_lines"]))

        if reason:
            st.caption(reason)
    else:
        score_col.caption("Not yet scored.")

    # Structured score (SCORING_MODE=structured) is kept in separate columns
    # so it can be compared against the raw score above for the same job.
    if pd.notna(structured_score_val):
        structured_breakdown_raw = sel.get("structured_score_breakdown", "") or ""
        structured_reason = sel.get("structured_score_reason", "") or ""

        structured_parsed = parse_score_breakdown(structured_breakdown_raw, fallback_score=structured_score_val)
        structured_computed = structured_parsed["computed_score"]
        st.markdown(f"**{_score_color(structured_computed)} Structured: {structured_computed} / 100**")

        if structured_parsed["items"]:
            for label, sc, mx, rsn in structured_parsed["items"]:
                line = f"**{label}**: {sc}/{mx}"
                if rsn:
                    line += f" — {rsn}"
                st.markdown(line)
        elif structured_parsed["legacy_lines"]:
            st.markdown("\n".join(f"- {p}" for p in structured_parsed["legacy_lines"]))

        if structured_reason:
            st.caption(structured_reason)

    job_id = sel.get("id", "")
    if btn_col.button("↺ Rescore", key=f"rescore_{job_id}", use_container_width=True):
        cand = get_candidate()
        summary = cand.get("summary", "")
        if not summary:
            st.warning("No candidate summary — generate one in the Profile tab first.")
        elif not (sel.get("description") or "").strip():
            st.warning("No description available for this job.")
        elif scoring_mode() == "structured":
            with st.spinner("Rescoring (structured)…"):
                requirements = parse_jd_extracted(sel.get("jd_extracted"))
                if requirements is None:
                    try:
                        req = extract_job_requirements(sel.get("description", ""), sel.get("company"))
                        requirements = req.model_dump()
                        update_job_fields(job_id, {"jd_extracted": json.dumps(requirements)})
                    except Exception as e:
                        st.error(f"Structured extraction failed: {e}")
                        requirements = None
                resume_profile = load_resume_profile(cand, log_fn=lambda msg: None) if requirements else None
                if requirements is not None and resume_profile is not None:
                    for result in score_jobs_structured(resume_profile, [{
                        "id": job_id, "requirements": requirements,
                        "is_remote": bool(sel.get("is_remote")),
                        "title": sel.get("title", ""), "company": sel.get("company", ""),
                    }]):
                        if result:
                            update_structured_scores(result)
            st.toast("Rescored (structured)!")
            st.rerun()
        else:
            job_dict = {
                "id": job_id,
                "title": sel.get("title", ""),
                "company": sel.get("company", ""),
                "description": sel.get("description", ""),
                "is_remote": sel.get("is_remote", False),
            }
            with st.spinner("Rescoring…"):
                for result in score_jobs(summary, [job_dict]):
                    if result:
                        update_scores(result)
            st.toast("Rescored!")
            st.rerun()
