"""ADGENCOV HTTP service (Phase C).

A thin FastAPI backend over the compiled C++ core and the GEO ingestion layer.
It accepts either an uploaded expression matrix or a GEO accession, runs the
recommender as an asynchronous job, and returns the same JSON shape that
:meth:`adgencov.AnalysisResult.to_dict` produces — the exact payload the web
and desktop GUIs (Phases D/F) consume.

Design boundary (unchanged from the rest of the project): Python owns the
network/parsing/orchestration; C++ owns the numerics.  This package adds only
request validation, an in-process job queue, and JSON marshalling.

Run it with::

    uvicorn adgencov.api:app --reload

or programmatically::

    from adgencov.api import create_app
    app = create_app()
"""
from __future__ import annotations

from .app import create_app
from .jobs import JobKind, JobRecord, JobState, JobStore

# A module-level app so ``uvicorn adgencov.api:app`` works out of the box.
app = create_app()

__all__ = ["app", "create_app", "JobStore", "JobRecord", "JobState", "JobKind"]
