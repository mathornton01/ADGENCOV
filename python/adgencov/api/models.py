"""Pydantic request/response schemas for the ADGENCOV service.

These models define the wire contract (and, via FastAPI, the auto-generated
OpenAPI docs at ``/docs``).  The analysis payload itself is intentionally left
as a free-form ``dict`` — it is produced verbatim by
:meth:`adgencov.AnalysisResult.to_dict`, and pinning its schema here would only
create a second source of truth to keep in sync with the core.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeParams(BaseModel):
    """Preprocessing + grouping + recommender knobs shared by both entry points.

    Mirrors the arguments of :func:`adgencov.geo.analyze_series` and the CLI so
    the API, GUI, and command line all expose the same controls.
    """

    n_genes: int = Field(500, ge=2, description="Top-variance genes to keep.")
    min_mean: float = Field(0.1, ge=0.0, description="Drop genes below this mean expression.")
    log_transform: bool = Field(True, description="Apply log2(x+1) before standardizing.")
    group: str = Field(
        "gene_family",
        description=(
            "Grouping strategy: gene_family, chromosome, reactome, go_process, "
            "custom, correlation_blocks, or hierarchical_wreath."
        ),
    )
    n_blocks: int = Field(4, ge=1, description="Number of blocks for correlation_blocks / wreath.")
    top_fraction: float = Field(
        0.01, gt=0.0, le=1.0, description="Fraction of gene pairs kept as network edges."
    )


class UploadParams(AnalyzeParams):
    """Extra knobs that only apply to an uploaded matrix (not GEO)."""

    sample_regex: str = Field(
        ".*",
        description="Regex selecting sample columns from the uploaded matrix header.",
    )
    gene_col: str = Field(
        "gene_short_name", description="Name of the gene-identifier column."
    )


class GeoAnalyzeRequest(AnalyzeParams):
    """Body for ``POST /analyze/geo`` — a GEO accession plus analysis knobs."""

    accession: str = Field(
        ...,
        min_length=3,
        description="GEO series accession, e.g. 'GSE52778'.",
        examples=["GSE52778"],
    )
    force: bool = Field(False, description="Bypass the on-disk GEO download cache.")


class JobSummary(BaseModel):
    """Lightweight job view returned by submit and list endpoints."""

    id: str
    kind: str
    state: str
    label: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


class JobDetail(JobSummary):
    """Full job view including the analysis payload once it has succeeded."""

    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None


class JobList(BaseModel):
    jobs: List[JobSummary]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "adgencov"
    version: str
    active_jobs: int
