"""Optional Langfuse tracing.

`observe` must exist before any submodule imports it, because they apply it as
an import-time decorator — that ordering constraint is why it lives in its own
module now rather than partway down `scanner/llm/__init__.py`.

Only litellm's own `langfuse_otel` callback is enabled; it ships spans over
OTLP/HTTP straight to Langfuse's OTel ingestion endpoint and never touches the
langfuse package's Python API, so this is insulated from that SDK's version drift.
"""
import os

import litellm

try:
    from langfuse import observe
    _LF_AVAILABLE = True
except ImportError:
    _LF_AVAILABLE = False

    def observe(_fn=None, **_kw):       # no-op decorator when langfuse isn't installed
        def _wrap(fn):
            return fn
        return _wrap(_fn) if _fn else _wrap


def is_tracing() -> bool:
    return _LF_AVAILABLE and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def install_callbacks() -> None:
    """Register litellm's Langfuse callbacks, if tracing is configured.

    Called once at package import; a no-op when tracing is off.
    """
    if is_tracing():
        litellm.success_callback = ["langfuse_otel"]
        litellm.failure_callback = ["langfuse_otel"]
