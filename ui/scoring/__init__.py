"""Scoring UI: the sidebar button, post-scan auto-scoring, and score display.

Split out of a single 415-line module whose three functions were 141, 139 and
102 lines. The two scoring runs now share `pipeline.run_scoring()` instead of
each carrying their own near-identical worker closure.

Re-exported here under their original names so callers are unaffected.
"""
from .auto_score import _auto_score_new
from .display import _render_score_display, _score_color
from .score_button import _render_score_button

__all__ = [
    "_auto_score_new",
    "_render_score_button",
    "_render_score_display",
    "_score_color",
]
