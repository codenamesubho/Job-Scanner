"""
Quick test: does litellm, pointed at a local cliproxyapi endpoint, support
Anthropic-style prompt caching?

Two scenarios:
    1. Plain litellm.completion() with a manual cache_control content block
       (generic litellm usage).
    2. The exact pattern this project uses in scanner/llm/raw_scoring.py —
       instructor.from_litellm() wrapping litellm.completion(), a plain
       string system message, and litellm's cache_control_injection_points
       auto-injection instead of a manual content block.

Usage:
    export CLAUDE_API_KEY="sk-..."                      # scanner/llm's own env var
    export CLIPROXY_MODEL="anthropic/claude-sonnet-4-6"  # model cliproxyapi expects
    python test_litellm_cliproxy_cache.py

What it does, per scenario:
    1. Sends a request with a large system prompt eligible for caching.
    2. Sends the SAME request again.
    3. Prints usage.cache_creation_input_tokens / cache_read_input_tokens from
       both calls. A cache hit on the 2nd call (cache_read_input_tokens > 0)
       confirms cliproxyapi is forwarding the cache directive through to
       Anthropic and returning cache usage fields litellm can parse.
"""

import os
import time

import instructor
import litellm
from pydantic import BaseModel

# scanner/llm/__init__.py's _PROVIDER_CONFIG["claude"] uses this exact
# api_base (bare, no /v1 — litellm's anthropic/ provider appends
# /v1/messages itself) and reads the key from CLAUDE_API_KEY.
API_BASE = os.environ.get("CLIPROXY_API_BASE", "http://localhost:8317")
API_KEY = os.environ.get("CLAUDE_API_KEY", os.environ.get("CLIPROXY_API_KEY", "sk-anything"))
MODEL = os.environ.get("CLIPROXY_MODEL", "anthropic/claude-sonnet-5")

# Needs to be long enough to clear Anthropic's minimum cacheable prefix
# (varies by model, roughly 1024-4096 tokens on older models). Repeating a
# paragraph is a cheap way to pad it out for this smoke test.
LARGE_SYSTEM_PROMPT = (
    "You are a helpful assistant embedded in a job-scanning tool. "
    "Answer tersely. " * 400
)


def _print_usage(label: str, response) -> object:
    usage = response.usage
    print(f"\n--- {label} ---")
    print("cache_creation_input_tokens:", getattr(usage, "cache_creation_input_tokens", None))
    print("cache_read_input_tokens:", getattr(usage, "cache_read_input_tokens", None))
    print("raw usage:", usage.model_dump() if hasattr(usage, "model_dump") else usage)
    return usage


def _check(usage) -> bool:
    return (getattr(usage, "cache_read_input_tokens", 0) or 0) > 0


# ── Scenario 1: plain litellm, manual cache_control content block ──────────

def run_plain_litellm(label: str):
    response = litellm.completion(
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        max_tokens=100,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": LARGE_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "In one sentence, what is your role?"},
        ],
    )
    print("reply:", response.choices[0].message.content)
    return _print_usage(label, response)


# ── Scenario 2: this project's actual pattern ───────────────────────────────
# Mirrors scanner/llm/__init__.py's _make_client(is_instructor=True) +
# scanner/llm/raw_scoring.py's _score_batch: instructor.from_litellm() over a
# litellm.completion partial bound to api_base/api_key, a plain string system
# message, and cache_control_injection_points instead of a manual content
# block — the project never hand-builds cache_control blocks itself.

class _Reply(BaseModel):
    role_summary: str


def run_project_pattern(label: str):
    completion_fn = litellm.completion
    client = instructor.from_litellm(
        lambda **kw: completion_fn(api_base=API_BASE, api_key=API_KEY, **kw),
        mode=instructor.Mode.JSON_SCHEMA,
    )
    last_raw_response = {}
    client.on("completion:response", lambda response: last_raw_response.__setitem__("value", response))

    result = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": LARGE_SYSTEM_PROMPT},
            {"role": "user", "content": "In one sentence, what is your role?"},
        ],
        cache_control_injection_points=[{"location": "message", "role": "system"}],
        response_model=_Reply,
        max_tokens=100,
        max_retries=1,
    )
    print("reply:", result.role_summary)
    return _print_usage(label, last_raw_response["value"])


if __name__ == "__main__":
    print("=" * 20, "Scenario 1: plain litellm + manual cache_control", "=" * 20)
    first = run_plain_litellm("request 1 (expect cache write)")
    time.sleep(1)
    second = run_plain_litellm("request 2 (expect cache read)")
    plain_ok = _check(second)

    print("\n" + "=" * 20, "Scenario 2: project's instructor.from_litellm() + cache_control_injection_points", "=" * 20)
    third = run_project_pattern("request 1 (expect cache write)")
    time.sleep(1)
    fourth = run_project_pattern("request 2 (expect cache read)")
    project_ok = _check(fourth)

    print("\n" + "=" * 60)
    print(f"Scenario 1 (plain litellm):        {'✅ cache hit' if plain_ok else '⚠️  no cache hit'}")
    print(f"Scenario 2 (project's own pattern): {'✅ cache hit' if project_ok else '⚠️  no cache hit'}")
    if not (plain_ok and project_ok):
        print(
            "\nIf either scenario didn't hit cache: check the prompt clears the "
            "model's minimum cacheable prefix, that cliproxyapi forwards the "
            "cache directive to Anthropic, and 'raw usage' above for how the "
            "response actually reports cache fields."
        )
