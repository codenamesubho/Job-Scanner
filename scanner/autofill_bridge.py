"""Bridge to the separate `Autofill-Job-Application` project — an LLM/
browser-use-driven agent that discovers *every* question on a real ATS
application form and answers it from a markdown candidate profile, with two
independent code-level guardrails (a restricted tool registry + a CDP-level
submit blocker) that make it structurally incapable of clicking submit.

This replaces `scanner/apply.py`'s keyword-heuristic engine (left in the repo
unused, not deleted) as the thing the UI's "Apply" button drives — the
heuristic only ever fills ~6 known fields and silently skips everything else
(custom screening questions, work-auth prompts, freeform "why us" boxes),
which is exactly the gap Autofill-Job-Application is built for.

Autofill-Job-Application is not a Job_Scanner dependency in requirements.txt
(it isn't published to PyPI) — it must be `pip install -e`d into this venv
from a local checkout; see CLAUDE.md. It drives its own async browser-use
agent loop, so this module shells out to its `autofill-fill` console script
via subprocess rather than importing it in-process — the same reasoning
scanner/apply.py uses to justify running Playwright's sync API on a
dedicated thread (see its module docstring): sidestepping any asyncio/
Streamlit-thread interaction entirely is simpler than managing it.
"""

import json
import os
import subprocess
import threading
from pathlib import Path

from . import database
from .profile import get_candidate, get_latest_resume


def context_path() -> Path:
    """`data/autofill_context.md`, resolved against the *current*
    `database.DB_PATH` rather than frozen at import time — DB_PATH is a
    rebindable module-level Path specifically so tests can point it at a
    tmp_path (see database.py's own docstring); freezing this at import
    time would silently keep pointing at the pre-test location."""
    return database.DB_PATH.parent / "autofill_context.md"


def _tmp_dir() -> Path:
    return database.DB_PATH.parent / "tmp"


def _runs_dir() -> Path:
    return database.DB_PATH.parent / "autofill_runs"

# How long to let one `autofill-fill` run go before giving up on it entirely
# (separate from --job-timeout/--batch-timeout, which bound the agent's own
# steps) — generous, since a real ATS form plus LLM round-trips can run
# several minutes.
_PROCESS_TIMEOUT_S = 600


class AutofillNotInstalled(RuntimeError):
    """Raised when the `autofill_job_application` package isn't importable."""


def ensure_installed() -> None:
    try:
        import autofill_job_application  # noqa: F401
    except ImportError as e:
        raise AutofillNotInstalled(
            "Autofill-Job-Application isn't installed in this environment. "
            "Run `pip install -e /path/to/Autofill-Job-Application` in this "
            "venv, then retry."
        ) from e


def _basics_lines(candidate: dict, extracted: dict) -> list[str]:
    lines = []
    name = candidate.get("name") or extracted.get("full_name")
    if name:
        lines.append(f"- Name: {name}")
    title = candidate.get("title") or extracted.get("current_title")
    if title:
        lines.append(f"- Current title: {title}")
    years_exp = candidate.get("years_exp") or extracted.get("years_exp")
    if years_exp:
        lines.append(f"- Years of experience: {years_exp}")
    location = extracted.get("location")
    if location:
        lines.append(f"- Location: {location}")
    for label, value in (
        ("Email", candidate.get("email")),
        ("Phone", candidate.get("phone")),
        ("LinkedIn", candidate.get("linkedin")),
        ("GitHub", extracted.get("github")),
    ):
        if value:
            lines.append(f"- {label}: {value}")
    return lines


def _experience_lines(extracted: dict) -> list[str]:
    lines = []
    skills = extracted.get("skills") or []
    if skills:
        lines.append(f"- Skills: {', '.join(skills)}")
    domains = extracted.get("domains") or []
    if domains:
        lines.append(f"- Domains: {', '.join(domains)}")
    education = extracted.get("education") or []
    for entry in education:
        lines.append(f"- Education: {entry}")
    work_summary = extracted.get("work_summary") or []
    if work_summary:
        lines.append("")
        lines.extend(f"- {entry}" for entry in work_summary)
    return lines


def build_context_markdown(candidate: dict | None = None, force: bool = False) -> Path:
    """Write `data/autofill_context.md` from the saved candidate profile.

    Only writes when the file doesn't already exist, unless `force=True`.
    Autofill-Job-Application treats this file as "the only source of truth"
    for the candidate — a user should be free to hand-edit it (add
    preferences, notes the extractor can't infer) without a later Apply
    click silently overwriting their edits. Deliberately omits
    salary/work-authorization/demographic fields even where the resume
    extraction has them — Autofill-Job-Application's own docs ask for those
    to be withheld from the candidate doc, since it never sends them to the
    model at all.
    """
    path = context_path()
    if path.exists() and not force:
        return path

    candidate = candidate if candidate is not None else get_candidate()
    extracted: dict = {}
    raw = candidate.get("resume_extracted")
    if raw:
        try:
            extracted = json.loads(raw)
        except (TypeError, ValueError):
            extracted = {}

    summary = candidate.get("summary") or ""

    parts = ["# About Me", "", "## Basics", *_basics_lines(candidate, extracted)]
    if summary:
        parts += ["", "## Summary", summary]
    experience = _experience_lines(extracted)
    if experience:
        parts += ["", "## Experience", *experience]
    parts += [
        "",
        "## Preferences",
        "<!-- Fill in manually: target roles, companies to avoid, work-arrangement "
        "preferences, etc. Salary expectations and work-authorization status are "
        "intentionally left out here — Autofill-Job-Application withholds those "
        "from the model; answer those questions yourself when the form asks. -->",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")
    return path


def write_resume_tempfile(resume: dict) -> Path:
    tmp_dir = _tmp_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / resume["filename"]
    path.write_bytes(resume["raw_content"])
    return path


def resolve_llm_env() -> dict:
    """Env-var overrides for the `autofill-fill` subprocess, derived from
    Job_Scanner's own Claude config so there's nothing new to set up by
    default. Returns a dict to merge into the subprocess environment —
    never mutates os.environ, so it can't leak into the rest of the running
    Streamlit process.

    Job_Scanner's Claude access is not a plain Anthropic API key against
    api.anthropic.com — CLAUDE_API_KEY/CLAUDE_MODEL route through a local
    CLIProxyAPI bridge (see scanner/llm/__init__.py's _PROVIDER_CONFIG) at
    http://localhost:8317, using litellm's "anthropic/<model>" naming.
    Autofill-Job-Application's own `litellm` provider mode speaks that same
    shape (model/api_base/api_key passed straight to litellm), so deriving
    AUTOFILL_LLM_PROVIDER=anthropic here would be wrong — it would try to
    hit api.anthropic.com directly with a key that's only valid against the
    local proxy.
    """
    overrides: dict = {}
    if os.getenv("AUTOFILL_LLM_API_KEY"):
        return overrides  # user has already configured it explicitly — leave it alone

    claude_key   = os.getenv("CLAUDE_API_KEY")
    claude_model = os.getenv("CLAUDE_MODEL")
    if not claude_key or not claude_model:
        return overrides  # nothing to derive from; let autofill-fill raise its own config error

    overrides["AUTOFILL_LLM_API_KEY"] = claude_key
    overrides["AUTOFILL_LLM_PROVIDER"] = "litellm"
    overrides["AUTOFILL_LLM_MODEL"] = f"anthropic/{claude_model}"
    overrides["AUTOFILL_LLM_BASE_URL"] = "http://localhost:8317"
    return overrides


def _latest_fill_run(out_dir: Path) -> dict | None:
    fills_dir = out_dir / "fills"
    if not fills_dir.exists():
        return None
    files = sorted(fills_dir.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def _summarize_fill_run(run: dict | None) -> dict:
    written, escalated, failed = [], [], []
    if run:
        for job in run.get("jobs", run.get("results", [])):
            for field in job.get("fields", []):
                label  = field.get("label") or field.get("name") or "field"
                status = field.get("write_status")
                if status == "written":
                    written.append(label)
                elif status == "escalated":
                    escalated.append(label)
                elif status == "failed":
                    failed.append(label)
    return {"filled": written, "escalated": escalated, "failed": failed}


def run_apply(job_url: str, log_fn=None,
              cancel_event: "threading.Event | None" = None) -> dict:
    """Launch `autofill-fill` against a single job URL. Never clicks submit
    (enforced inside Autofill-Job-Application itself, not here) — the
    headful Chrome window it drives is left open for manual review.

    Returns {"success", "filled", "escalated", "failed", "error"}.
    """
    def _log(msg: str) -> None:
        # Same convention as linkedin_playwright.py's _log/set_log_fn: print
        # to stdout by default, or route into a caller-supplied sink (the
        # UI's BackgroundJob.log) when one is given — so output is never
        # silently dropped when run_apply is called without UI wiring (a
        # script, a REPL, a future CLI entry point).
        if log_fn:
            log_fn(msg)
        else:
            print(msg, flush=True)

    try:
        ensure_installed()
    except AutofillNotInstalled as e:
        return {"success": False, "filled": [], "escalated": [], "failed": [], "error": str(e)}

    candidate = get_candidate()
    ctx_path = build_context_markdown(candidate)

    resume = get_latest_resume()
    resume_path = write_resume_tempfile(resume) if resume else None

    job_slug = "".join(c if c.isalnum() else "_" for c in job_url)[-60:]
    run_dir  = _runs_dir() / job_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = _tmp_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    urls_file = tmp_dir / f"{job_slug}.txt"
    urls_file.write_text(job_url + "\n")

    cmd = [
        "autofill-fill", str(urls_file),
        "--context", str(ctx_path),
        "--headful",
        "--out", str(run_dir),
    ]
    if resume_path:
        cmd += ["--resume", str(resume_path)]

    # Force the child's stdout unbuffered: `autofill-fill` is a Python console
    # script, and Python full-buffers stdout by default when it isn't attached
    # to a TTY (as here, piped through subprocess.Popen) — without this, log
    # lines sit in the child's own buffer instead of reaching `proc.stdout`
    # until it fills or the process exits, so the UI's live log box stays
    # empty for the whole run even though CLI use (a real TTY) looks fine.
    env = {**os.environ, **resolve_llm_env(), "PYTHONUNBUFFERED": "1"}

    _log(f"Launching autofill-fill for {job_url}…")
    try:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except FileNotFoundError:
        return {
            "success": False, "filled": [], "escalated": [], "failed": [],
            "error": "`autofill-fill` isn't on PATH — is Autofill-Job-Application installed "
                     "in this venv? (pip install -e /path/to/Autofill-Job-Application)",
        }

    for line in proc.stdout:
        _log(line.rstrip())
        if cancel_event is not None and cancel_event.is_set():
            proc.terminate()
            _log("Cancelled — the browser window (if opened) may still be open for manual review.")
            break

    try:
        returncode = proc.wait(timeout=_PROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        return {
            "success": False, "filled": [], "escalated": [], "failed": [],
            "error": f"Timed out after {_PROCESS_TIMEOUT_S}s.",
        }

    run = _latest_fill_run(run_dir)
    summary = _summarize_fill_run(run)

    if cancel_event is not None and cancel_event.is_set():
        return {"success": bool(summary["filled"]), "error": "Cancelled.", **summary}

    if returncode not in (0, 1):  # 2 = config error before Chrome even launched
        return {
            "success": False, "error": f"autofill-fill exited {returncode} (config error).",
            **summary,
        }

    success = bool(summary["filled"]) or bool(summary["escalated"])
    error = None if success else "No fields were filled or flagged — check the log above."
    return {"success": success, "error": error, **summary}
