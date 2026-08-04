"""A background worker plus its thread-safe progress state.

Streamlit widgets are not thread-safe, so anything long-running (a scan, a
scoring run) has to work the same way: spawn a daemon thread that writes only
into shared state, and let the main script thread read that state to update
progress bars and log boxes.

That pattern was hand-rolled three times — twice byte-identically in
ui/scoring.py and once per-source in ui/scan_handlers.py — as a bare
`threading.Lock()` next to an untyped dict, with ~30 open-coded `with lock:`
blocks reaching into it by string key. Nothing tied the lock to the data it
guarded, so forgetting to hold it was a silent race rather than an error.

`BackgroundJob` owns the lock, the state, and the thread together. Callers get
`log()`/`set()`/`snapshot()` and never touch the lock themselves.
"""
import threading
from dataclasses import dataclass, field, replace


@dataclass
class JobState:
    """A point-in-time view of a background job.

    Handed out only by `BackgroundJob.snapshot()`, which returns a copy — so
    callers can read it freely without holding the lock, and can't accidentally
    mutate the live state by writing to what they were given.
    """

    log: list[str] = field(default_factory=list)
    done: int = 0
    total: int = 0
    text: str = ""          # human-readable caption for the progress bar
    finished: bool = False
    error: str | None = None
    skip: str | None = None
    result: object = None

    @property
    def fraction(self) -> float:
        """Progress in 0..1, clamped — safe to hand straight to st.progress()."""
        if not self.total:
            return 0.0
        return min(self.done / self.total, 1.0)

    def log_text(self) -> str:
        return "\n".join(self.log)


class BackgroundJob:
    """Runs `target` on a daemon thread, exposing progress under one lock.

    The worker function is passed this object, so it reports progress through
    `log()`/`set()`/`add_done()` and checks `cancelled` — it never sees the lock.

    `finished` is set in a `finally`, so a worker that raises still terminates
    the job rather than leaving the UI polling forever; the exception text lands
    in `error`.
    """

    def __init__(self, name: str = "background-job"):
        self._lock = threading.Lock()
        self._state = JobState()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self.name = name

    # -- worker-side API ---------------------------------------------------

    def log(self, msg: str) -> None:
        with self._lock:
            self._state.log.append(msg)

    def set(self, **fields) -> None:
        """Assign one or more JobState fields. Unknown names raise, so a typo
        can't silently write a key nobody ever reads (the failure mode the old
        untyped dict had)."""
        with self._lock:
            for key, value in fields.items():
                if not hasattr(self._state, key):
                    raise AttributeError(f"JobState has no field {key!r}")
                setattr(self._state, key, value)

    def add_done(self, n: int) -> None:
        with self._lock:
            self._state.done += n

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def cancel_event(self) -> threading.Event:
        """The raw Event, for passing to scanner-layer functions that accept one."""
        return self._cancel

    # -- caller-side API ---------------------------------------------------

    def start(self, target) -> "BackgroundJob":
        """Run `target(self)` on a daemon thread. Returns self so callers can
        write `job = BackgroundJob().start(worker)`."""
        def _run() -> None:
            try:
                target(self)
            except Exception as e:
                self.set(error=str(e))
            finally:
                self.set(finished=True)

        self._thread = threading.Thread(target=_run, daemon=True, name=self.name)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()

    def snapshot(self) -> JobState:
        """A consistent copy of the current state, taken under the lock."""
        with self._lock:
            return replace(self._state, log=list(self._state.log))

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
