STATUSES      = ["new", "saved", "applied", "rejected"]
HOURS_OPTIONS = [24, 48, 72, 168, 336, 720]

# Match-score color thresholds (out of 100) shown next to a job's score.
SCORE_GOOD_THRESHOLD = 80
SCORE_OK_THRESHOLD   = 60
DEFAULT_MIN_SCORE    = 65

# Background-job polling: how often the main thread refreshes progress
# bars/log placeholders while a scan or scoring worker thread is running.
POLL_INTERVAL_S         = 0.4
SCORE_BUTTON_POLL_S     = 0.5  # _render_score_button ticks via st.rerun() instead of a blocking loop
LOG_TAIL_LINES          = 8    # single-source log placeholders (individual scans, score button)
AUTO_SCORE_LOG_TAIL_LINES = 6  # _auto_score_new's log placeholder
SCAN_ALL_LOG_TAIL_LINES = 4    # per-source log placeholders in "Scan All" (narrower, one of several)
