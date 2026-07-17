"""Generic "Apply and prefill" automation: launches a visible browser,
navigates to a job's application URL, and best-effort fills common form
fields (name, email, phone, LinkedIn, resume) from the saved candidate
profile.

Only text/email/tel/textarea/select/file inputs are ever queried or
interacted with. This module never clicks any button, submit input, or
role=button element, and never presses Enter inside a form — the browser
is always left open for the user to review and submit manually.
"""

import threading

from playwright.sync_api import sync_playwright

from .linkedin_playwright import SESSION_FILE as _LINKEDIN_SESSION_FILE
from .browser import launch_stealth_browser

# Only text-like inputs, textarea, select, and file inputs are ever queried —
# buttons, submit inputs, and role=button elements are structurally excluded
# so there is no code path that could ever hold a locator to a clickable
# submit-type control. Recurses into shadow roots: modern SDUI/web-component
# UIs (e.g. LinkedIn's Easy Apply modal) render their real form fields inside
# an open shadow root, which plain querySelectorAll does not pierce — without
# this, the scan sees the modal's visible text but zero actual inputs.
_SCAN_JS = """
() => {
    const results = [];
    let counter = 0;

    function buildBlob(el, root) {
        const parts = [];
        if (el.name) parts.push(el.name);
        if (el.id) parts.push(el.id);
        if (el.placeholder) parts.push(el.placeholder);
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) parts.push(ariaLabel);
        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy) {
            labelledBy.split(/\\s+/).forEach(id => {
                const t = root.getElementById ? root.getElementById(id) : null;
                if (t) parts.push(t.innerText || t.textContent || '');
            });
        }
        if (el.labels && el.labels.length) {
            for (const lab of el.labels) parts.push(lab.innerText || lab.textContent || '');
        }
        let blob = parts.join(' ').toLowerCase().trim();
        if (!blob) {
            const ancestor = el.closest('div, li, fieldset, section') || el.parentElement;
            if (ancestor) blob = (ancestor.innerText || '').slice(0, 80).toLowerCase().trim();
        }
        return blob;
    }

    function scan(root) {
        const els = root.querySelectorAll(
            "input:not([type=hidden]):not([type=submit]):not([type=button])" +
            ":not([type=checkbox]):not([type=radio]), textarea, select, input[type=file]"
        );
        for (const el of els) {
            const tagId = 'jsid-' + (counter++);
            el.setAttribute('data-jobscanner-id', tagId);
            results.push({
                tag_id: tagId,
                tag: el.tagName.toLowerCase(),
                type: (el.getAttribute('type') || el.tagName.toLowerCase()),
                blob: buildBlob(el, root),
            });
        }
        for (const el of root.querySelectorAll('*')) {
            if (el.shadowRoot) scan(el.shadowRoot);
        }
    }

    scan(document);
    return results;
}
"""

# (positive keywords, negative keywords) — a candidate matches a slot if its
# blob contains any positive keyword and none of the negative ones. Slots are
# tried in order and a matched element is removed from the pool so it can't
# double-match another slot.
_SLOTS: dict[str, tuple[list[str], list[str]]] = {
    "email":      (["email", "e-mail"], []),
    "phone":      (["phone", "mobile", "cell", "telephone"], []),
    "linkedin":   (["linkedin"], []),
    "first_name": (["first name", "given name", "fname"], ["company", "school", "reference", "emergency"]),
    "last_name":  (["last name", "surname", "family name", "lname"], ["company", "school", "reference", "emergency"]),
    "full_name":  (["full name", "your name", "applicant name", "name"],
                   ["company", "school", "reference", "emergency", "file"]),
}
_SLOT_ORDER = ["email", "phone", "linkedin", "first_name", "last_name", "full_name"]


def _match_slots(candidates: list[dict]) -> dict[str, str]:
    """Map scanned form fields to candidate-data slots. Pure function, no
    browser involved — unit-testable with static fixture dicts."""
    pool = [c for c in candidates if c.get("type") != "file"]
    # Prefer free-text elements over <select> dropdowns for these slots — a
    # phone/email/name value is rarely a valid option in an unrelated select
    # (e.g. LinkedIn's "phone country code" dropdown also contains "phone",
    # which would otherwise out-rank the real phone-number text input).
    pool.sort(key=lambda c: c.get("tag") == "select")
    matches: dict[str, str] = {}

    for slot in _SLOT_ORDER:
        positives, negatives = _SLOTS[slot]
        for cand in pool:
            blob = cand.get("blob", "")
            if any(neg in blob for neg in negatives):
                continue
            if any(pos in blob for pos in positives):
                matches[slot] = cand["tag_id"]
                pool.remove(cand)
                break

    file_candidates = [c for c in candidates if c.get("type") == "file"]
    resume_match = next(
        (c for c in file_candidates if "resume" in c.get("blob", "") or "cv" in c.get("blob", "")),
        None,
    )
    if not resume_match and len(file_candidates) == 1:
        resume_match = file_candidates[0]
    if resume_match:
        matches["resume"] = resume_match["tag_id"]

    return matches


_LLM_SLOTS = ("email", "phone", "linkedin", "first_name", "last_name", "full_name", "resume")


def _llm_match_slots(candidates: list[dict]) -> dict[str, str]:
    """LLM fallback for _match_slots — same {slot: tag_id} return shape, so
    it's a drop-in for the frame-scan loop. Called only when the keyword
    heuristic found nothing at all for a frame that has scannable fields,
    covering ATS platforms whose field naming a fixed keyword list can't
    anticipate. Degrades to {} on ANY failure (missing API key, rate limit,
    timeout, malformed output, breaker open, hallucinated/duplicate tag_id)
    — must never raise into _run_apply's scan loop.
    """
    try:
        from .llm import match_form_fields
        result = match_form_fields(candidates)
    except Exception:
        return {}

    valid_ids = {c["tag_id"] for c in candidates}
    used_ids: set[str] = set()
    matches: dict[str, str] = {}
    for slot in _LLM_SLOTS:
        tag_id = getattr(result, slot, None)
        if not tag_id or tag_id not in valid_ids or tag_id in used_ids:
            continue
        matches[slot] = tag_id
        used_ids.add(tag_id)
    return matches


def _candidate_values(candidate: dict) -> dict:
    name = (candidate.get("name") or "").strip()
    parts = name.split()
    return {
        "email": candidate.get("email") or "",
        "phone": candidate.get("phone") or "",
        "linkedin": candidate.get("linkedin") or "",
        "first_name": parts[0] if parts else "",
        "last_name": parts[-1] if len(parts) > 1 else "",
        "full_name": name,
    }


def _fill_frame(frame, matches: dict, values: dict, resume: dict | None) -> tuple[list[str], bool]:
    filled: list[str] = []
    resume_attached = False
    for slot, tag_id in matches.items():
        locator = frame.locator(f'[data-jobscanner-id="{tag_id}"]')
        if slot == "resume":
            if resume:
                try:
                    locator.set_input_files({
                        "name": resume["filename"],
                        "mimeType": resume.get("content_type") or "application/octet-stream",
                        "buffer": resume["raw_content"],
                    })
                    resume_attached = True
                    filled.append("resume")
                except Exception:
                    pass
            continue

        value = values.get(slot)
        if not value:
            continue
        try:
            tag_name = locator.evaluate("el => el.tagName.toLowerCase()")
            if tag_name == "select":
                try:
                    locator.select_option(label=value)
                except Exception:
                    continue
            else:
                locator.fill(value)
            filled.append(slot)
        except Exception:
            continue
    return filled, resume_attached


def _launch(p, *, headless: bool = True, storage_state_path: str | None = None):
    return launch_stealth_browser(
        p,
        headless=headless,
        storage_state_path=storage_state_path,
        viewport={"width": 1280, "height": 900},
    )


def _click_linkedin_apply(page, context, log_fn=None):
    """Click the real per-job Apply CTA on a LinkedIn job page (logged in
    via the saved session) and return whichever page ends up hosting the
    application form.

    Two distinct cases, both identified by aria-label (live-verified):
    - "LinkedIn Apply to this job" — in-house Easy Apply, opens a modal.
    - "Apply on company website" — external ATS (Greenhouse/Lever/Workday/
      etc.), href is a linkedin.com/safety/go/?url=<encoded target> redirect
      wrapper with no "/apply" substring, so matching on href is unreliable —
      aria-label alone is the stable signal. The job page also has "Apply"
      links on unrelated recommended-job cards elsewhere on the page; those
      have an empty aria-label, so they don't match either case.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    btn = page.query_selector('a[aria-label*="apply" i]')
    if not btn:
        _log("No Apply CTA found on the LinkedIn job page — scanning it as-is.")
        return page
    _log(f"Found Apply CTA: {btn.get_attribute('aria-label')!r} — clicking…")
    # A plain .click() is silently swallowed by LinkedIn's SPA event handling
    # here (confirmed live) — scrolling into view + a forced click is what
    # actually triggers the real click handler (same class of issue already
    # solved for send_linkedin_message's compose-button click).
    btn.scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    try:
        with context.expect_page(timeout=4000) as new_page_info:
            btn.click(force=True)
        new_page = new_page_info.value
        new_page.wait_for_load_state("domcontentloaded")
        new_page.wait_for_timeout(1000)
        _log(f"External site opened in a new tab: {new_page.url}")
        return new_page
    except Exception:
        # No new tab appeared — LinkedIn's own Easy Apply modal opened in place.
        # The SDUI-driven modal fetches its form config asynchronously, so give
        # it a few seconds to finish rendering before scanning for fields.
        _log("No new tab — assuming LinkedIn's own Easy Apply modal opened in place.")
        page.wait_for_timeout(4000)
        return page


def apply_and_prefill(url: str, log_fn=None) -> dict:
    """Launch a visible browser, navigate to `url`, and best-effort fill
    application-form fields from the saved candidate profile + latest resume.
    Never clicks any button, submit input, or role=button element — the
    browser is left open for the user to review and submit manually.

    If given, log_fn(str) is called with step-by-step progress messages
    ("Launching browser…", "Heuristic matched: email, phone…", etc.) — same
    log_fn convention as scanner.llm.score_jobs. log_fn runs on the internal
    worker thread described below, so callers updating Streamlit widgets
    from it must only append to a locked shared list/dict, never touch
    st.* directly (same rule as every other background-thread caller in
    this codebase).

    Returns {"success", "filled_fields", "resume_attached", "error"}.

    Runs the whole Playwright session in a dedicated worker thread and blocks
    until filling is done. This is necessary, not just style: Playwright's
    sync API refuses a second sync_playwright().start() on a thread that
    already has one active (confirmed live — "using Playwright Sync API
    inside the asyncio loop"), so calling this twice in a row from the same
    calling thread (e.g. Streamlit's script-run thread, reused across
    reruns) would crash the second time if the first browser is still open
    for review. A fresh thread per call sidesteps that entirely — the
    long-lived "keep open until closed" part continues on this same worker
    thread after the caller gets its result back.
    """
    result: dict = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            result.update(_run_apply(url, log_fn))
        except Exception as e:
            if log_fn:
                log_fn(f"Failed: {e}")
            result.update({"success": False, "filled_fields": [], "resume_attached": False, "error": str(e)})
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()
    if not done.wait(timeout=75):
        if log_fn:
            log_fn("Timed out after 75s.")
        return {
            "success": False, "filled_fields": [], "resume_attached": False,
            "error": "Timed out opening the application (75s). The browser may still be loading.",
        }
    return result


def _run_apply(url: str, log_fn=None) -> dict:
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    from .profile import get_candidate, get_latest_resume

    candidate = get_candidate() or {}
    resume = get_latest_resume()
    values = _candidate_values(candidate)

    is_linkedin = "linkedin.com/jobs/view" in url
    storage_state_path = (
        str(_LINKEDIN_SESSION_FILE)
        if is_linkedin and _LINKEDIN_SESSION_FILE.exists()
        else None
    )

    _log("Launching browser…")
    # Use start()/stop() directly (not the `with` context manager) so the
    # driver survives past this function's return — a daemon thread keeps
    # the browser open until the user closes it (same pattern as
    # linkedin_playwright.send_linkedin_message(auto_send=False)).
    pw = sync_playwright().start()
    browser, context = _launch(pw, headless=False, storage_state_path=storage_state_path)

    def _cleanup() -> None:
        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass

    try:
        page = context.new_page()
        _log(f"Navigating to {url}…")
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(1500)

        target_page = page
        if is_linkedin:
            _log("LinkedIn job page detected — resolving the real Apply target…")
            target_page = _click_linkedin_apply(page, context, log_fn)

        filled_fields: list[str] = []
        resume_attached = False
        llm_attempted = False
        for frame in target_page.frames:
            try:
                candidates = frame.evaluate(_SCAN_JS)
            except Exception:
                continue
            if not candidates:
                continue
            _log(f"Scanning frame {frame.url!r} — {len(candidates)} candidate field(s).")
            matches = _match_slots(candidates)
            if matches:
                _log(f"Heuristic matched: {', '.join(matches)}.")
            if not matches and not llm_attempted:
                # Cap the LLM fallback to one attempt per run — real ATS pages
                # often have several zero-field frames (ad iframes, chat
                # widgets) that would otherwise each burn a call.
                llm_attempted = True
                _log("Heuristic found nothing — trying the LLM fallback…")
                matches = _llm_match_slots(candidates)
                if matches:
                    _log(f"LLM matched: {', '.join(matches)}.")
                else:
                    _log("LLM fallback found nothing either.")
            if not matches:
                continue
            filled_fields, resume_attached = _fill_frame(frame, matches, values, resume)
            if filled_fields:
                break

        success = bool(filled_fields)
        if success:
            resume_note = " + resume" if resume_attached else ""
            _log(f"Filled {len(filled_fields)} field(s){resume_note}: {', '.join(filled_fields)}.")
        else:
            _log("Couldn't auto-detect form fields — leaving browser open for manual apply.")
        error = None if success else "Couldn't auto-detect form fields — browser opened for manual apply."

        def _wait_for_close() -> None:
            try:
                browser.wait_for_event("disconnected", timeout=3_600_000)
            except Exception:
                pass
            finally:
                _cleanup()

        threading.Thread(target=_wait_for_close, daemon=True).start()
        _log("Browser left open for review — submit manually when ready.")

        return {
            "success": success,
            "filled_fields": filled_fields,
            "resume_attached": resume_attached,
            "error": error,
        }
    except Exception as e:
        _log(f"Failed: {e}")
        _cleanup()
        return {"success": False, "filled_fields": [], "resume_attached": False, "error": str(e)}
