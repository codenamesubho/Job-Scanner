from pydantic import BaseModel, ConfigDict, Field

from . import execute_with_breaker, observe

_REFERRAL_PROMPT = (
    "You are writing a short LinkedIn message from a job seeker to a connection, asking for "
    "a referral. The message MUST do both of these, in order:\n"
    "1. One brief sentence tying the candidate's background to the role — a specific, "
    "concrete reason they're relevant, not a generic recap of the whole summary.\n"
    "2. An explicit, direct ask: request that the contact refer them for the role, or put in "
    "a good word / point them to the right person if a direct referral isn't possible.\n"
    "Keep the whole message under 130 words, warm but professional. If a job URL is provided, "
    "include it naturally near the ask. Output only the message text — no subject line, no "
    "greeting label, no sign-off placeholder like '[Your Name]'.\n\n"
    "Candidate summary: {summary}\n"
    "Contact: {contact_name}, {contact_title} at {company}\n"
    "Role applying for: {job_title} at {company}\n"
    "Job URL: {job_url}\n"
)

_REFERRAL_RETRY_NUDGE = (
    "\nYour previous attempt was too short or didn't clearly ask for a referral. Write a "
    "complete message that still does both required things above.\n"
)

_FORM_FIELD_MATCH_PROMPT = """\
You are matching job-application form fields on a webpage to a candidate's saved profile \
data slots. Below is a list of form fields scanned from the page, each with a tag_id, its \
HTML tag/input type, and a short "blob" of text scraped from its name/id/placeholder/\
aria-label/associated <label> text (lowercased).

Fields:
{fields_block}

Candidate data slots to fill, each needs AT MOST ONE field's tag_id (or none if no field \
clearly matches):
- email: the applicant's email address
- phone: the applicant's phone number
- linkedin: the applicant's LinkedIn profile URL
- first_name: the applicant's first/given name (a field asking ONLY for the first name)
- last_name: the applicant's last/family/surname (a field asking ONLY for the last name)
- full_name: the applicant's full name as one combined field (only if there is a single \
combined name field — do not use this if separate first/last name fields already cover it)
- resume: a file-upload field for the resume/CV document

Rules:
- Only map a field to a slot if the field's blob CLEARLY indicates it collects that exact \
piece of data. If you are not confident, leave the slot null — do not guess.
- Never map a field whose blob refers to something else, even if loosely related: company \
name, school/education, reference contact, emergency contact, cover letter, message/notes, \
salary/compensation expectation, recruiter/referrer name, portfolio/website (unless it \
explicitly says linkedin), veteran/disability/EEO status, or any other unrelated question.
- A field with tag=select (a dropdown) is almost never the right match for email/phone/name/\
linkedin — e.g. a "phone country code" dropdown also contains the word "phone" but is NOT \
the phone-number field. Only map a select field if its blob unmistakably means the exact \
data requested (this will be rare to never for these slots).
- Each field's tag_id may be used for AT MOST ONE slot — never reuse the same tag_id twice.
- Only output tag_ids that appear in the Fields list above — never invent one.
- If no field clearly matches a slot, leave that slot null (JSON null) rather than picking \
the closest-but-imperfect option.

Respond with ONLY a JSON object of exactly this form (JSON null, not the string "null", for \
any slot with no confident match):
{{"email": "<tag_id or null>", "phone": "<tag_id or null>", "linkedin": "<tag_id or null>", \
"first_name": "<tag_id or null>", "last_name": "<tag_id or null>", \
"full_name": "<tag_id or null>", "resume": "<tag_id or null>"}}
"""


class FormFieldMap(BaseModel):
    """One tag_id per candidate-data slot the LLM confidently matched among a
    scanned list of form fields. Any slot with no confident match is left
    null — the model must not guess by elimination."""
    model_config = ConfigDict(extra="ignore")

    email: str | None = Field(default=None)
    phone: str | None = Field(default=None)
    linkedin: str | None = Field(default=None)
    first_name: str | None = Field(default=None)
    last_name: str | None = Field(default=None)
    full_name: str | None = Field(default=None)
    resume: str | None = Field(default=None)


_REFERRAL_MIN_WORDS = 15
_REFERRAL_ASK_KEYWORDS = (
    "refer", "referral", "recommend", "good word", "vouch", "connect me",
    "point me", "introduce me", "put me in touch",
)


def _looks_like_an_ask(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _REFERRAL_ASK_KEYWORDS)


@observe(name="draft_referral_message")
def draft_referral_message(candidate_summary: str, contact: dict, job: dict) -> str:
    job_url = job.get("job_url_direct") or job.get("job_url") or ""
    prompt = _REFERRAL_PROMPT.format(
        summary=candidate_summary[:1000],
        contact_name=contact.get("name", ""),
        contact_title=contact.get("title", ""),
        company=job.get("company", ""),
        job_title=job.get("title", ""),
        job_url=job_url or "not provided",
    )

    def _query(client, model):
        max_tokens = 300
        current_prompt = prompt
        for attempt in range(2):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": current_prompt}],
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            truncated = getattr(choice, "finish_reason", None) == "length"
            degenerate = len(text.split()) < _REFERRAL_MIN_WORDS or not _looks_like_an_ask(text)
            if not truncated and not degenerate:
                return text
            if attempt == 0:
                max_tokens = min(max_tokens * 2, 1000)
                # Truncated output just needs more room; a short/ask-less
                # output needs the nudge — don't tell a too-long response it
                # was "too short."
                if degenerate:
                    current_prompt = prompt + _REFERRAL_RETRY_NUDGE
                continue
            if not text:
                raise ValueError("Referral draft came back empty after retry.")
            return text  # best effort — degenerate/possibly-truncated, but non-empty

    # Pinned to Gemini rather than the default provider: confirmed by a live
    # call in this environment that the default (Claude via the local
    # CLIProxyAPI proxy) currently 404s on the configured CLAUDE_MODEL, while
    # Gemini is the path everything else (extraction, and scoring's fallback)
    # already relies on successfully. Revisit this once Claude/CLIProxyAPI is
    # confirmed healthy — there's no cost rationale for pinning a low-volume,
    # quality-sensitive call like this one, unlike extract_job_requirements's
    # deliberate Gemini pin.
    return execute_with_breaker(_query, provider_override="gemini")


def _format_field_line(c: dict) -> str:
    blob = (c.get("blob") or "").replace('"', "'")
    return f'- tag_id={c.get("tag_id")} tag={c.get("tag")} type={c.get("type")} blob="{blob}"'


def match_form_fields(candidates: list[dict]) -> FormFieldMap:
    """Ask the LLM to map scanned job-application form fields to candidate-
    data slots — fallback for scanner.apply's keyword heuristic when it
    finds nothing on an unfamiliar ATS platform. Raises on ANY failure
    (breaker open, missing API key, rate limit, timeout, malformed output)
    — the caller (scanner.apply._llm_match_slots) is responsible for
    catching and degrading gracefully.
    """
    fields_block = "\n".join(_format_field_line(c) for c in candidates)
    prompt = _FORM_FIELD_MATCH_PROMPT.format(fields_block=fields_block)

    def _query(client, model):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=FormFieldMap,
            max_tokens=300,
            max_retries=1,
            timeout=15,
        )

    return execute_with_breaker(_query, is_instructor=True)
