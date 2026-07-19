from instructor.core import IncompleteOutputException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import execute_with_breaker, observe

_SUMMARY_PROMPT = (
    "Based on the following resume text, write a concise professional summary "
    "in first person. Highlight key skills, years of experience, "
    "and career focus. Highlight my work domain, tech stack and the years I have spent working on them."
    "Output only the summary text — no headings or labels.\n\n"
    "Resume:\n{resume_text}"
)


class JobRequirements(BaseModel):
    """Structured extraction of a raw job description (see
    extract_job_requirements) — the only place raw JD text is read end to
    end. Powers structured scoring (score_jobs_structured) so later scoring
    calls never re-send raw text."""
    model_config = ConfigDict(extra="ignore")

    must_haves: list[str] = Field(default_factory=list)
    nice_to_haves: list[str] = Field(default_factory=list)
    min_yoe: int | None = Field(default=None)
    max_yoe: int | None = Field(default=None)
    seniority_band: str | None = Field(default=None)   # intern/junior/mid/senior/staff/principal/director+
    location: str | None = Field(default=None)
    remote_policy: str | None = Field(default=None)     # remote/hybrid/onsite/unspecified
    work_auth: str | None = Field(default=None)         # e.g. "no sponsorship", "citizenship required"
    tech_stack: list[str] = Field(default_factory=list)
    company_type: str | None = Field(default=None)      # e.g. "big tech", "startup", "agency"
    company_size: str | None = Field(default=None)      # e.g. "1-50", "51-500", "500+"
    industry_domain: str | None = Field(default=None)
    key_responsibilities: list[str] = Field(default_factory=list)
    description: str | None = Field(
        default=None,
        description="A short, consistently-styled synopsis (2-4 plain-prose sentences) of the "
                    "role's core expectations and day-to-day responsibilities — not a copy of "
                    "the raw posting, and not company boilerplate (benefits, EEO statements, "
                    "culture blurbs, legal text). Written in the same length/tone/structure for "
                    "every job regardless of how differently formatted or verbose the original "
                    "source text was, so it reads consistently across jobs from different "
                    "sources (LinkedIn, Greenhouse, Lever, Ashby, JSearch, etc.) and is directly "
                    "comparable to a candidate's per-role resume_profile.work_summary entries "
                    "when matching a resume against this job.",
    )


class ResumeProfile(BaseModel):
    """Structured extraction of a candidate's resume text (see
    extract_resume_profile) — feeds structured scoring and, for the
    fill-in-application fields, a future apply.py form-fill data source."""
    model_config = ConfigDict(extra="ignore")

    # apply.py form-fill slots — field names match scanner/apply.py's _SLOTS
    full_name: str | None = Field(
        default=None,
        description="Candidate's full name, almost always the very first line of the resume "
                    "(the header), often in a larger font/all-caps — extract it even without an "
                    "explicit 'Name:' label.",
    )
    first_name: str | None = Field(
        default=None,
        description="Given name. If not stated as its own field, split it from full_name.",
    )
    last_name: str | None = Field(
        default=None,
        description="Family name. If not stated as its own field, split it from full_name.",
    )
    email: str | None = Field(default=None)
    phone: str | None = Field(default=None)
    linkedin: str | None = Field(default=None)
    github: str | None = Field(default=None)

    # scoring-relevant fields, mirroring JobRequirements' shape
    years_exp: float | None = Field(default=None)
    seniority_band: str | None = Field(default=None)
    skills: list[str] = Field(default_factory=list)
    skill_years: dict[str, float] = Field(
        default_factory=dict,
        description="Approximate total years of hands-on experience per skill/technology named "
                    "in `skills`, computed from the dated work experience entries where that "
                    "skill was actually used (e.g. a skill used in a role from 2019-2022 and "
                    "again 2023-Present contributes ~4 years) — not from how prominently it's "
                    "listed in a standalone 'Skills' section. Overlapping/concurrent roles that "
                    "use the same skill should not be double-counted. Only include a skill here "
                    "if the resume's dated work history actually shows it being used; a skill "
                    "listed only in a bare skills list with no traceable dated usage can still "
                    "go in `skills` but should be omitted from this mapping.",
    )
    current_title: str | None = Field(
        default=None,
        description="Candidate's current or most recent job title/designation. Usually printed "
                    "right under or beside the name in the header (e.g. 'Jane Doe — Senior "
                    "Backend Engineer'); if the header has no title, use the job title of the "
                    "most recent (topmost/undated 'Present') entry in the work experience "
                    "section instead of leaving this null.",
    )
    current_company_type: str | None = Field(
        default=None,
        description="Category of the candidate's current/most recent employer (e.g. 'big tech', "
                    "'startup', 'agency', 'consultancy') inferred from that employer's name/"
                    "description — not the employer's name itself.",
    )
    work_auth: str | None = Field(default=None)
    location: str | None = Field(default=None)
    domains: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    work_summary: list[str] = Field(
        default_factory=list,
        description="One short summary per work experience entry, same order as listed on the "
                    "resume, each prefixed with company, title, AND the role's date range exactly "
                    "as written on the resume (e.g. 'Acme Corp — Senior Backend Engineer (Nov "
                    "2022 - Present): ...'). The date range is required, not optional — it's what "
                    "lets skill recency be judged later (a skill used in the most recent role "
                    "reads very differently from the same skill only appearing in a role from "
                    "several years ago). Cover whichever of these the resume actually describes "
                    "for that role: the technology/stack used, the architecture or system worked "
                    "on (scale, design, service boundaries), and the specific problem/challenge "
                    "tackled — skip whichever of the three isn't described rather than padding "
                    "with generic filler.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_alternate_shapes(cls, data):
        """extract_resume_profile runs under instructor's MD_JSON mode (a
        prompted-JSON mode, not tool-calling), so the model is free to
        organize richer resumes into nested groupings instead of this flat
        schema — e.g. a "contact": {...} object instead of top-level
        email/phone/linkedin, or a "work_experience": [{company, title,
        work_summary}, ...] array instead of the flat `work_summary` list.
        Without this, `extra="ignore"` silently drops those mismatched keys
        and the fields just come back null/empty — not a validation error,
        so it's easy to mistake for the model failing to find the data at
        all. Normalize the common alternate shapes here before validation.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        contact = data.pop("contact", None)
        if isinstance(contact, dict):
            for key in ("email", "phone", "linkedin", "github", "location", "full_name"):
                if not data.get(key) and contact.get(key):
                    data[key] = contact[key]

        work_experience = data.pop("work_experience", None)
        if isinstance(work_experience, list) and not data.get("work_summary"):
            summaries = []
            for entry in work_experience:
                if not isinstance(entry, dict):
                    continue
                company = entry.get("company") or ""
                prefix = " — ".join(
                    p for p in (company, entry.get("title")) if p
                )
                date_range = " - ".join(
                    str(p) for p in (entry.get("start_date"), entry.get("end_date")) if p
                )
                if date_range:
                    prefix = f"{prefix} ({date_range})" if prefix else f"({date_range})"
                summary = entry.get("work_summary") or entry.get("summary") or ""
                # The model is instructed to prefix work_summary text with
                # "Company — Title (dates): ..." itself, so when it also
                # nests entries under "work_experience" it sometimes embeds
                # that same prefix inside the per-entry summary text too —
                # don't prepend our own computed prefix on top of that.
                already_prefixed = bool(company) and company in summary[:len(prefix) + 20]
                if already_prefixed or not prefix:
                    summaries.append(summary)
                else:
                    summaries.append(f"{prefix}: {summary}")
            if summaries:
                data["work_summary"] = summaries

        education = data.get("education")
        if isinstance(education, list):
            flattened = []
            for entry in education:
                if isinstance(entry, str):
                    flattened.append(entry)
                elif isinstance(entry, dict):
                    parts = [entry.get("degree"), entry.get("field"), entry.get("institution")]
                    date_range = "-".join(
                        str(p) for p in (entry.get("start_date"), entry.get("end_date")) if p
                    )
                    if date_range:
                        parts.append(f"({date_range})")
                    if entry.get("gpa"):
                        parts.append(f"GPA: {entry['gpa']}")
                    flattened.append(", ".join(p for p in parts if p))
            if flattened:
                data["education"] = flattened

        # The model reliably returns full_name but often skips splitting it
        # into first_name/last_name even though the field descriptions ask
        # for it — split deterministically here rather than depending on
        # the model to do it consistently.
        full_name = data.get("full_name")
        if isinstance(full_name, str) and full_name.strip():
            parts = full_name.split()
            if not data.get("first_name") and parts:
                data["first_name"] = parts[0]
            if not data.get("last_name") and len(parts) > 1:
                data["last_name"] = " ".join(parts[1:])

        return data


@observe(name="generate_summary")
def generate_summary(resume_text: str) -> str:
    def _query(client, model):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _SUMMARY_PROMPT.format(resume_text=resume_text[:8000])}],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    return execute_with_breaker(_query)


_JD_EXTRACT_MAX_CHARS = 100000

_JD_EXTRACT_PROMPT = """\
Extract structured information from the following raw job description. Only use information \
explicitly present in the text — leave a field null/empty if it isn't stated; do not guess or \
infer facts not present.

Also write `description`: a short synopsis (2-4 plain-prose sentences) covering only the role's \
core expectations and day-to-day responsibilities — skip benefits, EEO statements, culture blurbs, \
and other company boilerplate that isn't specific to what this person would actually do. Keep the \
same length and tone regardless of how long, short, or differently formatted this particular \
posting is, so descriptions read consistently across jobs from different sources.

For `company_type`/`company_size`: most job descriptions don't state these explicitly. The \
hiring company's name is given below as "Company". If you recognize this company from your own \
general knowledge (e.g. well-known big tech, a large/well-funded startup or unicorn, a well-known \
but smaller/early-stage startup), use that knowledge to fill company_type/company_size — the same \
way an experienced recruiter would recognize a company by name — even though that knowledge isn't \
in the JD text itself. Only leave these null if the company name is generic/unrecognizable and \
the JD text also says nothing about company size/stage.

Company: {company}

Job Description:
{description}
"""


@observe(name="extract_job_requirements")
def extract_job_requirements(description: str, company: str | None = None) -> JobRequirements:
    """The only function that reads a raw job description end to end. Always
    uses Gemini (provider_override="gemini") with the cheap "extract" model
    class (GEMINI_EXTRACT_MODEL) regardless of LLM_PROVIDER — Gemini is the
    designated JD extraction provider while Claude handles resume extraction
    and scoring. `company` is the job's company name (not JD text) — passed
    through so the model can use its own general knowledge of named companies
    to fill company_type/company_size, the same way score_jobs()'s raw path
    recognizes company names directly."""
    prompt = _JD_EXTRACT_PROMPT.format(description=description, company=company or "Not specified")

    def _query(client, model):
        max_tokens = 6000
        for attempt in range(2):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=JobRequirements,
                    max_tokens=max_tokens,
                    max_retries=2,
                    timeout=45,
                )
            except IncompleteOutputException:
                if attempt == 0:
                    max_tokens *= 2
                else:
                    raise

    return execute_with_breaker(_query, is_instructor=True, model_class="extract",
                                provider_override="gemini")


_RESUME_EXTRACT_PROMPT = """\
Extract structured candidate information from the following resume text. Only use information \
explicitly present in the text — leave a field null/empty if it isn't stated; do not guess or \
infer facts not present.

The resume's header (its first few lines) is the primary source for full_name and current_title, \
even when it has no labels like "Name:" — treat a standalone line at the very top, often in a \
larger font or all-caps, as the name. A short line right after it (or next to it, or separated by \
a pipe/dash from the name) is usually the current title/designation. If the header has no title, \
fall back to the job title of the most recent (topmost, or marked "Present") entry in the work \
experience section — don't leave current_title null just because it's absent from the header.

For phone/email/linkedin/github, extract exactly as written (don't reformat). For each work \
experience entry, generate one short summary in work_summary, PREFIXED with company, title, and \
that role's date range exactly as written on the resume (e.g. "Acme Corp — Senior Backend \
Engineer (Nov 2022 - Present): ..." or "(2019-2022): ..." if company/title aren't stated) — the \
date range is required on every entry, since it's how a later comparison can tell whether a skill \
was used recently or only in an old role. After the date-range prefix, cover whichever of the \
technology/stack used, the architecture or system worked on (scale, design, service boundaries), \
and the specific problem/challenge that role tackled the resume actually describes; don't invent \
or pad with generic filler for whichever of those three isn't stated for that role.

Duration matters as much as the words used, so read the work experience section's dated entries \
(not just a standalone "Skills" list) to fill skill_years: for each skill/technology actually used \
in a dated role, add up the years of the date range(s) where it was used (e.g. "used in a role \
from 2019-2022" ≈ 3 years); if the same skill recurs in a later role, add that range too, but \
don't double-count time from concurrent/overlapping roles. A skill only ever mentioned in a bare \
skills list, with no traceable dated role behind it, still belongs in `skills` but should be left \
out of skill_years rather than guessed at. This also feeds years_exp and seniority_band — base \
those on the full span of dated work experience, not just the most recent role.

Resume:
{resume_text}
"""


@observe(name="extract_resume_profile")
def extract_resume_profile(resume_text: str) -> ResumeProfile:
    """Structured extraction of resume text. Uses the "structured_score"
    (Sonnet-class) model tier rather than the cheap "extract" tier used by
    extract_job_requirements — there's only one resume per candidate (versus
    many JDs), so it's worth spending more on getting it right."""
    prompt = _RESUME_EXTRACT_PROMPT.format(resume_text=resume_text)

    def _query(client, model):
        max_tokens = 10000  # skill_years/work_summary (one entry per role) can push a
                             # multi-role resume's output well past what a short profile needs
        for attempt in range(3):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=ResumeProfile,
                    max_tokens=max_tokens,
                    max_retries=2,
                    timeout=45,
                )
            except IncompleteOutputException:
                if attempt < 2:
                    max_tokens *= 2
                else:
                    raise

    return execute_with_breaker(_query, is_instructor=True, model_class="structured_score")
