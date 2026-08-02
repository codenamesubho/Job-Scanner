from pydantic import BaseModel, ConfigDict, Field

from . import execute_with_breaker, observe
from ._prompt_loader import load_prompt_file

# Prompt text lives in scanner/llm/prompts/referral.json (versioned per entry).
_PROMPTS = load_prompt_file("referral")
_REFERRAL_PROMPT = _PROMPTS["referral_message"]["template"]
_REFERRAL_RETRY_NUDGE = _PROMPTS["referral_retry_nudge"]["template"]
_FORM_FIELD_MATCH_PROMPT = _PROMPTS["form_field_match"]["template"]


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
        max_tokens = 400  # comfortable headroom for a 250-word reply on Flash-Lite (no reasoning-token overhead)
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

    # Pinned to Gemini (confirmed live: the default Claude path via the local
    # CLIProxyAPI proxy 404s on the configured CLAUDE_MODEL). Uses its own
    # "referral" model tier — REQUIRES GEMINI_REFERRAL_MODEL=gemini-2.5-flash-
    # lite in .env (a pinned version, not the gemini-flash-latest alias).
    # Without that env var set, this silently falls back to GEMINI_MODEL
    # (gemini-flash-latest) via _model_class_override, which is the exact
    # model confirmed live to spend ~800 tokens/call on invisible internal
    # "thinking" before writing anything visible — reproducing the original
    # mid-sentence-truncation bug with NO error or warning. Flash-Lite does
    # no such reasoning pass — confirmed live at 165-210 completion tokens
    # for a complete, on-ask reply to the same prompt. If Claude is ever
    # revisited here, use claude-sonnet-4-6 (confirmed working live), not
    # Haiku (confirmed live to refuse the task, self-identifying as "Claude
    # Code" — the same CLIProxyAPI persona-bridging issue documented in
    # _query's messages= comment above) — and note _looks_like_an_ask()
    # below is a keyword check, not a refusal detector: Haiku's refusal text
    # contained "referral"/"reference the role" and would have passed it,
    # shipping the refusal as a draft undetected.
    return execute_with_breaker(_query, provider_override="gemini", model_class="referral")


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
            max_tokens=1000,
            max_retries=1,
            timeout=15,
        )

    return execute_with_breaker(_query, is_instructor=True)
