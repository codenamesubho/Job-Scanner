"""Tests for ui.background.BackgroundJob.

This class replaced a hand-rolled lock + untyped dict that appeared three times.
Its whole value is the synchronization invariant, so that is what is tested:
snapshots must be consistent and isolated, and a job must always terminate.
"""
import threading
import time

import pytest

from ui.background import BackgroundJob, JobState


def _drain(job, timeout=5.0):
    """Wait for a job to finish, failing loudly rather than hanging forever."""
    deadline = time.time() + timeout
    while job.is_alive():
        if time.time() > deadline:
            pytest.fail("background job did not finish")
        time.sleep(0.01)
    return job.snapshot()


def test_runs_target_and_reports_progress():
    def worker(job):
        job.set(total=2, text="working")
        job.log("first")
        job.add_done(1)
        job.log("second")
        job.add_done(1)
        job.set(result="done-value")

    snap = _drain(BackgroundJob().start(worker))

    assert snap.log == ["first", "second"]
    assert (snap.done, snap.total) == (2, 2)
    assert snap.result == "done-value"
    assert snap.finished is True
    assert snap.error is None


def test_a_raising_worker_still_finishes_and_records_the_error():
    """The UI polls until `finished`; a worker that dies without setting it
    would hang the page forever."""
    def worker(job):
        raise RuntimeError("worker exploded")

    snap = _drain(BackgroundJob().start(worker))

    assert snap.finished is True
    assert snap.error == "worker exploded"


def test_snapshot_is_isolated_from_later_writes():
    """Callers read the snapshot without the lock, so it must not alias live state."""
    job = BackgroundJob()
    job.log("before")
    snap = job.snapshot()
    job.log("after")

    assert snap.log == ["before"]
    snap.log.append("mutating the copy")
    assert job.snapshot().log == ["before", "after"]


def test_set_rejects_unknown_fields():
    """The old untyped dict silently accepted a typo'd key that nothing read."""
    with pytest.raises(AttributeError):
        BackgroundJob().set(scored=5)


def test_cancel_is_visible_to_the_worker():
    started = threading.Event()

    def worker(job):
        started.set()
        while not job.cancelled:
            time.sleep(0.01)
        job.set(result="stopped cleanly")

    job = BackgroundJob().start(worker)
    started.wait(timeout=5)
    job.cancel()

    assert _drain(job).result == "stopped cleanly"


def test_concurrent_logging_from_several_threads_loses_nothing():
    """The lock exists for exactly this; without it appends interleave badly."""
    job = BackgroundJob()

    def spam():
        for _ in range(200):
            job.log("x")

    threads = [threading.Thread(target=spam) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(job.snapshot().log) == 800


# ------------------------------------------------------------------- JobState

@pytest.mark.parametrize("done, total, expected", [
    (0, 0, 0.0),       # nothing known yet — must not divide by zero
    (0, 10, 0.0),
    (5, 10, 0.5),
    (10, 10, 1.0),
    (11, 10, 1.0),     # clamped: st.progress rejects >1.0
])
def test_fraction_is_clamped_and_zero_safe(done, total, expected):
    assert JobState(done=done, total=total).fraction == expected


def test_log_text_joins_lines():
    assert JobState(log=["a", "b"]).log_text() == "a\nb"


# ------------------------------------- ui.scan_handlers._start_source_job

def test_start_source_job_wires_runner_output_into_the_job():
    """Covers the parallel-scan worker without touching the network: the runner
    is a stub, so this checks only the plumbing (log sink, progress mapping,
    ScanResult capture) that _run_parallel_sources' polling loop reads back."""
    from scanner import ScanResult
    from ui.scan_handlers import _start_source_job

    def fake_runner(criteria, log_fn, progress_fn):
        log_fn("searching…")
        progress_fn(0.5, "halfway")
        return ScanResult(found=7, new=3)

    snap = _drain(_start_source_job("Fake Source", fake_runner, criteria=None))

    assert snap.log == ["searching…"]
    assert snap.text == "halfway"
    assert snap.fraction == 0.5
    assert snap.result == ScanResult(found=7, new=3)
    assert snap.error is None


def test_start_source_job_records_a_failing_source_without_killing_the_batch():
    from ui.scan_handlers import _start_source_job

    def boom(criteria, log_fn, progress_fn):
        raise RuntimeError("source is down")

    snap = _drain(_start_source_job("Broken", boom, criteria=None))

    assert snap.error == "source is down"
    assert snap.finished is True
    assert snap.result is None
