"""
Quick test: does litellm, pointed at a local cliproxyapi endpoint, support
Anthropic-style prompt caching?

Usage:
    export CLIPROXY_API_BASE="http://localhost:8317"   # your cliproxyapi URL
    export CLIPROXY_API_KEY="sk-..."                    # cliproxyapi key, if it requires one
    export CLIPROXY_MODEL="anthropic/claude-sonnet-4-6" # model name cliproxyapi expects
    python test_litellm_cliproxy_cache.py

What it does:
    1. Sends a request with a large system prompt marked with cache_control
       (ephemeral) so it's eligible for caching.
    2. Sends the SAME request again.
    3. Prints usage.cache_creation_input_tokens / cache_read_input_tokens from
       both calls. A cache hit on the 2nd call (cache_read_input_tokens > 0)
       confirms cliproxyapi is forwarding cache_control through to Anthropic
       and returning cache usage fields litellm can parse.
"""

import os
import time

import litellm

API_BASE = os.environ.get("CLIPROXY_API_BASE", "http://localhost:8317")
API_KEY = os.environ.get("CLIPROXY_API_KEY", "sk-anything")
MODEL = os.environ.get("CLIPROXY_MODEL", "anthropic/claude-sonnet-4-6")

# Needs to be long enough to clear Anthropic's minimum cacheable prefix
# (varies by model, roughly 1024-4096 tokens on older models). Repeating a
# paragraph is a cheap way to pad it out for this smoke test.
LARGE_SYSTEM_PROMPT = (
    "You are a helpful assistant embedded in a job-scanning tool. "
    "Answer tersely. " * 400
)


def run_once(label: str):
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

    usage = response.usage
    print(f"\n--- {label} ---")
    print("reply:", response.choices[0].message.content)
    print("prompt_tokens:", usage.prompt_tokens)
    print("cache_creation_input_tokens:", getattr(usage, "cache_creation_input_tokens", None))
    print("cache_read_input_tokens:", getattr(usage, "cache_read_input_tokens", None))
    # Full usage dict as a fallback, in case the fields above are named
    # differently for this litellm/provider version.
    print("raw usage:", usage.model_dump() if hasattr(usage, "model_dump") else usage)
    return usage


if __name__ == "__main__":
    first = run_once("request 1 (expect cache write)")
    time.sleep(1)
    second = run_once("request 2 (expect cache read)")

    cache_read = getattr(second, "cache_read_input_tokens", 0) or 0
    if cache_read > 0:
        print("\n✅ Caching works: second request read from cache.")
    else:
        print(
            "\n⚠️  No cache read detected on the second request. "
            "Either the prompt is below the model's minimum cacheable size, "
            "cliproxyapi isn't forwarding cache_control, or usage fields "
            "aren't being parsed — check 'raw usage' above."
        )
