import json
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt_file(module_name: str) -> dict:
    """Load scanner/llm/prompts/<module_name>.json — a dict of
    {key: {"version": ..., "template": ...}} entries. Called once per
    module at import time; each module pulls its own templates out into
    its existing constant names, so everything downstream (.format() calls,
    function bodies) is unaffected by prompt text living in JSON instead of
    inline Python strings."""
    path = _PROMPTS_DIR / f"{module_name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)
