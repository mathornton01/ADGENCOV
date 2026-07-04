"""An in-process asynchronous job store for analysis runs.

The recommender is CPU-bound and can take seconds to tens of seconds on a real
GEO series, so the API returns immediately with a job id and runs the work on a
background thread pool.  Because the heavy numerics live in the C++ core (which
releases the GIL), a :class:`~concurrent.futures.ThreadPoolExecutor` gives real
parallelism without a separate worker process — the right weight for a single
scientific service.

This deliberately stays in-memory: no Redis, no Celery.  A future turn can swap
:class:`JobStore` for a durable backend behind the same interface if the hosted
portal needs persistence across restarts.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class JobState(str, Enum):
    """Lifecycle of an analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobKind(str, Enum):
    """What produced the job — an uploaded matrix or a GEO accession."""

    UPLOAD = "upload"
    GEO = "geo"


@dataclass
class JobRecord:
    """A single analysis job and its outcome."""

    id: str
    kind: JobKind
    label: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    state: JobState = JobState.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Live progress for the status bar: fraction in [0, 1] and a human phase.
    progress: float = 0.0
    phase: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "state": self.state.value,
            "label": self.label,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "progress": self.progress,
            "phase": self.phase,
        }

    def detail(self) -> Dict[str, Any]:
        d = self.summary()
        d["params"] = self.params
        d["result"] = self.result
        return d


class JobStore:
    """Thread-safe registry of :class:`JobRecord` backed by a thread pool."""

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="adgencov-job"
        )
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._futures: Dict[str, Future] = {}

    # -- submission ---------------------------------------------------------
    def submit(
        self,
        kind: JobKind,
        fn: Callable[[Callable[[float, str], None]], Dict[str, Any]],
        *,
        label: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        """Register a job and schedule *fn* to run on the pool.

        *fn* takes a single ``progress`` callback — ``progress(fraction, phase)``
        with ``fraction`` in ``[0, 1]`` and a short human-readable ``phase`` — and
        returns the JSON-serializable analysis dict (typically
        ``AnalysisResult.to_dict()``).  Calling the callback updates the live
        job state that clients poll for a status bar; passing it is optional (a
        job that never calls it simply reports 0 until it finishes).  Any
        exception *fn* raises is captured and surfaced as ``state=failed`` with
        the message in ``error``.
        """
        job = JobRecord(id=uuid.uuid4().hex, kind=kind, label=label, params=params or {})
        with self._lock:
            self._jobs[job.id] = job
            fut = self._pool.submit(self._run, job.id, fn)
            self._futures[job.id] = fut
        return job

    def _set_progress(self, job_id: str, fraction: float, phase: str) -> None:
        """Record live progress for *job_id* (thread-safe; ignored if gone)."""
        frac = 0.0 if fraction < 0.0 else 1.0 if fraction > 1.0 else float(fraction)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress = frac
            job.phase = phase

    def _run(
        self,
        job_id: str,
        fn: Callable[[Callable[[float, str], None]], Dict[str, Any]],
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:  # removed before it started
                return
            job.state = JobState.RUNNING
            job.started_at = time.time()

        def progress(fraction: float, phase: str) -> None:
            self._set_progress(job_id, fraction, phase)

        try:
            result = fn(progress)
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job.result = result
                job.state = JobState.SUCCEEDED
                job.finished_at = time.time()
                job.progress = 1.0
                job.phase = "Complete"
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = JobState.FAILED
                job.finished_at = time.time()

    # -- access -------------------------------------------------------------
    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def remove(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            fut = self._futures.pop(job_id, None)
        if fut is not None:
            fut.cancel()
        return job is not None

    def active_count(self) -> int:
        with self._lock:
            return sum(
                1
                for j in self._jobs.values()
                if j.state in (JobState.PENDING, JobState.RUNNING)
            )

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Optional[JobRecord]:
        """Block until *job_id* finishes (test/CLI convenience).

        Returns the (terminal) record, or ``None`` if the job is unknown.
        """
        with self._lock:
            fut = self._futures.get(job_id)
        if fut is not None:
            try:
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001 - outcome is recorded on the job
                pass
        return self.get(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
