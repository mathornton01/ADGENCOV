"""The FastAPI application factory for the ADGENCOV service.

Endpoints
---------
``GET  /health``               liveness + version + active-job count
``POST /analyze/upload``       multipart matrix upload → job id
``POST /analyze/geo``          JSON {accession, ...params} → job id
``GET  /search/geo``           term search → matching GEO series (GSE…)
``POST /translate/proteins``   protein/gene ids → protein names (UniProt)
``GET  /jobs``                 list all jobs (newest first)
``GET  /jobs/{id}``            job detail (result present once succeeded)
``DELETE /jobs/{id}``          forget/cancel a job

Both analyze endpoints are asynchronous: they validate the request, enqueue a
job, and return ``202 Accepted`` with a :class:`JobSummary`.  Clients poll
``GET /jobs/{id}`` until ``state`` is ``succeeded`` (payload in ``result``) or
``failed`` (message in ``error``).  Interactive docs live at ``/docs``.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# The single-page dashboard (Phase D) ships as static files next to this
# package; the API serves it directly so the whole product is one deployable.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

from .. import __version__
from .jobs import JobKind, JobStore
from .models import (
    GeoAnalyzeRequest,
    GeoSearchResponse,
    HealthResponse,
    JobDetail,
    JobList,
    JobSummary,
    ProteinTranslateRequest,
    ProteinTranslateResponse,
    UploadParams,
)
from .service import run_geo_analysis, run_upload_analysis

# Origins allowed to call the API from a browser.  The hosted portal lives on
# thorntonstatistical.com; localhost entries cover local GUI development.
DEFAULT_ALLOWED_ORIGINS = [
    "https://thorntonstatistical.com",
    "https://www.thorntonstatistical.com",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]

# Guard rail so an over-large upload can't exhaust memory before we ever parse
# it.  50 MB comfortably covers a dense genes-by-samples matrix.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def create_app(
    store: Optional[JobStore] = None,
    *,
    allowed_origins: Optional[list] = None,
    max_workers: int = 2,
) -> FastAPI:
    """Build and return a configured :class:`fastapi.FastAPI` instance.

    A fresh :class:`JobStore` is created unless one is injected (handy for
    tests that want to drive jobs synchronously via :meth:`JobStore.wait`).
    """
    app = FastAPI(
        title="ADGENCOV Service",
        version=__version__,
        summary="Algebraic-Diversity Genetic Covariance estimation for transcriptomics.",
        description=(
            "Upload an expression matrix or point at a GEO accession; the "
            "service runs the leave-one-out estimator recommender in the C++ "
            "core and returns the recommendation, covariance edges, and gene "
            "blocks as JSON."
        ),
    )
    app.state.store = store if store is not None else JobStore(max_workers=max_workers)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins if allowed_origins is not None else DEFAULT_ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    def get_store(request: Request) -> JobStore:
        return request.app.state.store

    # -- health -------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(store: JobStore = Depends(get_store)) -> HealthResponse:
        return HealthResponse(version=__version__, active_jobs=store.active_count())

    # -- upload -------------------------------------------------------------
    @app.post(
        "/analyze/upload",
        response_model=JobSummary,
        status_code=202,
        tags=["analyze"],
    )
    async def analyze_upload(
        store: JobStore = Depends(get_store),
        file: UploadFile = File(..., description="Expression matrix (TSV/CSV)."),
        n_genes: int = Form(150),
        min_mean: float = Form(0.1),
        log_transform: bool = Form(True),
        group: str = Form("gene_family"),
        n_blocks: int = Form(4),
        top_fraction: float = Form(0.01),
        sample_regex: str = Form(".*"),
        gene_col: str = Form("gene_short_name"),
    ) -> JobSummary:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
            )
        try:
            params = UploadParams(
                n_genes=n_genes,
                min_mean=min_mean,
                log_transform=log_transform,
                group=group,
                n_blocks=n_blocks,
                top_fraction=top_fraction,
                sample_regex=sample_regex,
                gene_col=gene_col,
            )
        except ValueError as exc:  # pydantic validation
            raise HTTPException(status_code=422, detail=str(exc))

        label = file.filename or "upload"
        job = store.submit(
            JobKind.UPLOAD,
            lambda progress: run_upload_analysis(raw, params, progress=progress),
            label=label,
            params=params.model_dump(),
        )
        return JobSummary(**job.summary())

    # -- geo ----------------------------------------------------------------
    @app.post(
        "/analyze/geo",
        response_model=JobSummary,
        status_code=202,
        tags=["analyze"],
    )
    def analyze_geo(
        req: GeoAnalyzeRequest,
        store: JobStore = Depends(get_store),
    ) -> JobSummary:
        job = store.submit(
            JobKind.GEO,
            lambda progress: run_geo_analysis(req, progress=progress),
            label=req.accession,
            params=req.model_dump(),
        )
        return JobSummary(**job.summary())

    # -- GEO term search ----------------------------------------------------
    @app.get("/search/geo", response_model=GeoSearchResponse, tags=["discover"])
    def search_geo_endpoint(
        term: str,
        retmax: int = 20,
    ) -> GeoSearchResponse:
        """Find GEO series (GSE…) matching a free-text query via NCBI E-utilities."""
        from ..bioquery import BioQueryError, search_geo

        if not term or not term.strip():
            raise HTTPException(status_code=422, detail="query 'term' is required")
        try:
            hits = search_geo(term, retmax=retmax)
        except BioQueryError as exc:
            raise HTTPException(status_code=502, detail=f"GEO search failed: {exc}")
        return GeoSearchResponse(
            term=term.strip(),
            count=len(hits),
            hits=[h.to_dict() for h in hits],
        )

    # -- protein-id translation --------------------------------------------
    @app.post("/translate/proteins", response_model=ProteinTranslateResponse, tags=["discover"])
    def translate_proteins_endpoint(
        req: ProteinTranslateRequest,
    ) -> ProteinTranslateResponse:
        """Translate protein/gene identifiers to protein names via UniProt."""
        from ..bioquery import BioQueryError, translate_protein_ids

        try:
            names = translate_protein_ids(
                req.ids, source=req.source, reviewed_only=req.reviewed_only
            )
        except BioQueryError as exc:
            raise HTTPException(status_code=502, detail=f"translation failed: {exc}")
        results = [n.to_dict() for n in names]
        return ProteinTranslateResponse(
            count=len(results),
            matched=sum(1 for r in results if r["matched"]),
            results=results,
        )

    # -- jobs ---------------------------------------------------------------
    @app.get("/jobs", response_model=JobList, tags=["jobs"])
    def list_jobs(store: JobStore = Depends(get_store)) -> JobList:
        return JobList(jobs=[JobSummary(**j.summary()) for j in store.list()])

    @app.get("/jobs/{job_id}", response_model=JobDetail, tags=["jobs"])
    def get_job(job_id: str, store: JobStore = Depends(get_store)) -> JobDetail:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        return JobDetail(**job.detail())

    @app.delete("/jobs/{job_id}", status_code=204, tags=["jobs"])
    def delete_job(job_id: str, store: JobStore = Depends(get_store)) -> None:
        if not store.remove(job_id):
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")

    # -- dashboard (static SPA) --------------------------------------------
    # Mounted LAST so every explicit API route above (and FastAPI's own
    # /docs, /openapi.json) takes precedence; the mount then serves index.html
    # at "/" and the JS/CSS assets.  html=True makes "/" resolve to index.html.
    if os.path.isdir(STATIC_DIR):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")

    return app
