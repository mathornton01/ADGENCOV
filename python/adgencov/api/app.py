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
import re
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
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
    InteractionsResponse,
    JobDetail,
    JobList,
    JobSummary,
    MultiAnalyzeRequest,
    ProteinTranslateRequest,
    ProteinTranslateResponse,
    SymbolTranslateRequest,
    SymbolTranslateResponse,
    UploadParams,
)
from .service import (
    run_combine_analysis,
    run_compare_analysis,
    run_geo_analysis,
    run_upload_analysis,
)

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


def _figure_overrides(request) -> dict:
    """Turn ?figsize=12,8&show_hubs=false&... into FigureConfig overrides.

    Only fields FigureConfig actually declares are accepted, and each is coerced
    to that field's type, so the URL cannot inject arbitrary state.
    """
    from dataclasses import fields as _fields

    from ..figure import FigureConfig

    spec = {f.name: f for f in _fields(FigureConfig)}
    out: dict = {}
    for key, raw in request.query_params.items():
        f = spec.get(key)
        if f is None:
            continue
        ann = str(f.type)
        low = raw.strip().lower()
        try:
            if "bool" in ann:
                out[key] = low in ("1", "true", "yes", "on")
            elif key == "figsize":
                a, b = raw.split(",")
                out[key] = (float(a), float(b))
            elif "Sequence[str]" in ann:
                out[key] = [v.strip() for v in raw.split(",") if v.strip()]
            elif "Sequence[float]" in ann:
                out[key] = [float(v) for v in raw.split(",") if v.strip()]
            elif "int" in ann:
                out[key] = None if low in ("", "none") else int(raw)
            elif "float" in ann:
                out[key] = None if low in ("", "none") else float(raw)
            else:
                out[key] = raw
        except (TypeError, ValueError):
            raise HTTPException(status_code=422,
                                detail=f"bad value for figure option {key!r}: {raw!r}")
    return out


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

    # Make browsers revalidate the SPA assets on every load (they still get a
    # fast 304 when unchanged). Without this, a browser can serve a stale app.js
    # against fresh index.html after a deploy and render a broken/blank page.
    @app.middleware("http")
    async def _revalidate_static(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

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

    # -- multi-dataset: combine / compare -----------------------------------
    @app.post("/analyze/combine", response_model=JobSummary, status_code=202, tags=["analyze"])
    def analyze_combine(
        req: MultiAnalyzeRequest,
        store: JobStore = Depends(get_store),
    ) -> JobSummary:
        """Pool several GEO series into one matrix and run a single analysis.

        Only genes shared by every series are kept, and each gene is standardized
        within its own dataset before pooling, so the covariance reflects
        within-study co-variation rather than which study a sample came from.
        """
        job = store.submit(
            JobKind.COMBINE,
            lambda progress: run_combine_analysis(req, progress=progress),
            label="+".join(req.accessions),
            params=req.model_dump(),
        )
        return JobSummary(**job.summary())

    @app.post("/analyze/compare", response_model=JobSummary, status_code=202, tags=["analyze"])
    def analyze_compare(
        req: MultiAnalyzeRequest,
        store: JobStore = Depends(get_store),
    ) -> JobSummary:
        """Analyze several GEO series separately and compare the results.

        Reports which estimator each dataset selects, pairwise top-edge overlap
        (Jaccard) and sign agreement, and the edges recovered in more than one
        dataset.
        """
        job = store.submit(
            JobKind.COMPARE,
            lambda progress: run_compare_analysis(req, progress=progress),
            label=" vs ".join(req.accessions),
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

    # -- gene id -> symbol translation -------------------------------------
    @app.post("/translate/symbols", response_model=SymbolTranslateResponse, tags=["discover"])
    def translate_symbols_endpoint(
        req: SymbolTranslateRequest,
    ) -> SymbolTranslateResponse:
        """Batch-resolve gene ids to current official (HGNC) symbols via mygene.info."""
        from ..bioquery import BioQueryError, translate_gene_symbols

        try:
            syms = translate_gene_symbols(req.ids, species=req.species)
        except BioQueryError as exc:
            raise HTTPException(status_code=502, detail=f"symbol translation failed: {exc}")
        results = [s.to_dict() for s in syms]
        return SymbolTranslateResponse(
            count=len(results),
            matched=sum(1 for r in results if r["matched"]),
            results=results,
        )

    # -- STRING interaction search -----------------------------------------
    @app.get("/interactions", response_model=InteractionsResponse, tags=["discover"])
    def interactions_endpoint(
        genes: str,
        species: str = "9606",
        limit: int = 10,
        required_score: int = 400,
    ) -> InteractionsResponse:
        """Look up STRING-db interaction partners for one or two genes.

        *genes* is a comma-separated list (a heatmap cell supplies its row+column
        gene).  Returns each gene's strongest partners plus the direct pair score.
        """
        from ..bioquery import BioQueryError
        from ..stringdb import interactions

        ids = [g.strip() for g in (genes or "").split(",") if g.strip()]
        if not ids:
            raise HTTPException(status_code=422, detail="query 'genes' is required")
        if len(ids) > 2:
            raise HTTPException(status_code=422, detail="supply at most two genes")
        try:
            data = interactions(
                ids, species=species, limit=limit, required_score=required_score
            )
        # Catch BioQueryError, not just StringError: StringError *subclasses* it,
        # so the transport layer's plain BioQueryError (e.g. STRING answering 404
        # for ids it doesn't know, such as miRNAs) escaped and became a 500.
        except BioQueryError as exc:
            raise HTTPException(status_code=502, detail=f"interaction lookup failed: {exc}")
        return InteractionsResponse(**data)

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

    # -- exports -------------------------------------------------------------
    # Publication-ready downloads rendered by adgencov.export — the same module
    # the manuscript scripts use, so a table downloaded here is identical to the
    # one in the paper.
    EXPORTS = {
        "table.tex": ("application/x-tex", "ranking_to_latex"),
        "table.csv": ("text/csv", "ranking_to_csv"),
        "edges.csv": ("text/csv", "edges_to_csv"),
        "blocks.csv": ("text/csv", "blocks_to_csv"),
        "covariance.csv": ("text/csv", "covariance_to_csv"),
        "figure.pdf": ("application/pdf", None),
        "figure.png": ("image/png", None),
        "compare.csv": ("text/csv", "compare_to_csv"),
        "compare.tex": ("application/x-tex", "compare_to_latex"),
    }

    @app.get("/jobs/{job_id}/export/{artifact}", tags=["jobs"])
    def export_job(job_id: str, artifact: str, request: Request,
                   store: JobStore = Depends(get_store)):
        """Download a finished analysis as LaTeX or CSV.

        *artifact* is one of figure.pdf, figure.png, table.tex, table.csv,
        edges.csv, blocks.csv, covariance.csv (single analyses) or compare.csv,
        compare.tex (a compare run).

        Figure exports accept any ``FigureConfig`` field as a query parameter,
        e.g. ``figure.png?figsize=12,8&show_hubs=false&community_colors=#111,#666``.
        """
        from .. import export as exporters

        if artifact not in EXPORTS:
            raise HTTPException(
                status_code=404,
                detail=f"unknown export {artifact!r}; try one of {sorted(EXPORTS)}",
            )
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        detail = job.detail()
        if detail.get("state") != "succeeded" or not detail.get("result"):
            raise HTTPException(
                status_code=409,
                detail=f"job {job_id!r} is {detail.get('state')}; exports need a succeeded job",
            )

        media, fn_name = EXPORTS[artifact]
        payload = detail["result"]
        is_compare = "comparison" in payload

        if fn_name is None:                      # figure.pdf / figure.png
            from ..figure import FigureUnavailable, render_network
            if is_compare:
                raise HTTPException(
                    status_code=409,
                    detail="figure export applies to a single analysis, not a compare run",
                )
            try:
                body = render_network(payload, fmt=artifact.rsplit(".", 1)[1],
                                      **_figure_overrides(request))
            except FigureUnavailable as exc:
                raise HTTPException(status_code=501, detail=str(exc))
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", (job.label or job_id))[:60]
            return Response(content=body, media_type=media,
                            headers={"Content-Disposition":
                                     f'attachment; filename="adgencov_{stem}_{artifact}"'})

        if artifact.startswith("compare.") != is_compare:
            kind = "a compare run" if is_compare else "a single analysis"
            raise HTTPException(
                status_code=409,
                detail=f"{artifact} does not apply to {kind}",
            )
        kwargs = {}
        if fn_name in ("ranking_to_latex", "ranking_to_csv"):
            kwargs["criterion"] = (detail.get("params") or {}).get("criterion", "loo")
        if fn_name in ("ranking_to_latex", "compare_to_latex"):
            kwargs["caption"] = f"ADGENCOV estimator ranking ({job.label})"
        try:
            body = getattr(exporters, fn_name)(payload, **kwargs)
        except ValueError as exc:                 # e.g. covariance omitted
            raise HTTPException(status_code=409, detail=str(exc))
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", (job.label or job_id))[:60]
        return Response(
            content=body,
            media_type=media,
            headers={"Content-Disposition":
                     f'attachment; filename="adgencov_{stem}_{artifact}"'},
        )

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
