"""Pydantic request/response schemas for the ADGENCOV service.

These models define the wire contract (and, via FastAPI, the auto-generated
OpenAPI docs at ``/docs``).  The analysis payload itself is intentionally left
as a free-form ``dict`` — it is produced verbatim by
:meth:`adgencov.AnalysisResult.to_dict`, and pinning its schema here would only
create a second source of truth to keep in sync with the core.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Allowed values, kept in one place so the wire contract can't drift from the
# core.  These mirror adgencov.ESTIMATOR_FAMILIES / AD_MODES and the groupings
# adgencov._core.build_group_labels accepts (plus "auto").
_VALID_GROUPS = frozenset({
    "none", "gene_family", "chromosome", "reactome", "go_process",
    "correlation_blocks", "hierarchical_wreath", "custom", "custom_group_map",
    "auto",
})
_VALID_FAMILIES = frozenset({
    "sample", "ridge", "lasso", "elastic_net", "ledoit_wolf", "oas",
})
_VALID_AD_MODES = frozenset({"none", "projection", "target", "optimal"})
_ACCESSION_RE = re.compile(r"^GSE\d+(-GPL\d+)?$", re.IGNORECASE)


class AnalyzeParams(BaseModel):
    """Preprocessing + grouping + recommender knobs shared by both entry points.

    Mirrors the arguments of :func:`adgencov.geo.analyze_series` and the CLI so
    the API, GUI, and command line all expose the same controls.
    """

    n_genes: int = Field(150, ge=2, description="Top-variance genes to keep.")
    min_mean: float = Field(0.1, ge=0.0, description="Drop genes below this mean expression.")
    log_transform: bool = Field(True, description="Apply log2(x+1) before standardizing.")
    group: str = Field(
        "gene_family",
        description=(
            "Symmetry grouping for Algebraic Diversity: gene_family, chromosome, "
            "reactome, go_process, custom, correlation_blocks, hierarchical_wreath, "
            "or 'auto' to try several built-in structures and keep the best."
        ),
    )
    n_blocks: int = Field(4, ge=1, description="Number of blocks for correlation_blocks / wreath.")
    families: Optional[List[str]] = Field(
        None,
        description=(
            "Estimator families to score: any of sample, ridge, lasso, "
            "elastic_net, ledoit_wolf, oas. Null = all (run every estimator). "
            "Give one for a single-estimator run."
        ),
    )
    ad_modes: Optional[List[str]] = Field(
        None,
        description=(
            "Algebraic-Diversity variants to include: 'none' (ordinary), "
            "'projection' (hard Reynolds projection), 'target' (Eq. 2 convex "
            "shrinkage). Null = all three. Use ['none'] to turn AD off."
        ),
    )
    sweep: bool = Field(
        True,
        description="Sweep each estimator's hyper-parameters (vs a single default).",
    )
    top_fraction: float = Field(
        0.01, gt=0.0, le=1.0, description="Fraction of gene pairs kept as network edges."
    )
    cv_folds: Optional[int] = Field(
        None,
        ge=2,
        description=(
            "Estimator-scoring cross-validation. When null (default) uses exact "
            "leave-one-out CV. When set to an integer k, uses k-fold CV instead — "
            "n/k times fewer fits, so much faster on large sample counts, at a "
            "slight cost in selection precision (the recommended estimator is "
            "usually unchanged). Typical fast value: 10."
        ),
    )
    criterion: str = Field(
        "loo",
        pattern="^(loo|ebic|kfold)$",
        description=(
            "Model-selection criterion for ranking the estimator grid: 'loo' "
            "(exact leave-one-out CV, default), 'kfold' (k-fold CV, cv_folds "
            "folds), or 'ebic' (one-pass Extended BIC — ~n times faster than "
            "leave-one-out — using ebic_gamma). Runs server-side per dataset."
        ),
    )
    ebic_gamma: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Extended BIC penalty in [0, 1], used only when criterion='ebic' "
            "(0 = ordinary BIC; larger is more conservative for p >> n)."
        ),
    )

    @field_validator("group")
    @classmethod
    def _check_group(cls, v: str) -> str:
        if v not in _VALID_GROUPS:
            raise ValueError(f"group must be one of {sorted(_VALID_GROUPS)}, got {v!r}")
        return v

    @field_validator("families")
    @classmethod
    def _check_families(cls, v):
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("families cannot be an empty list; use null for all")
        bad = [f for f in v if f not in _VALID_FAMILIES]
        if bad:
            raise ValueError(
                f"unknown estimator families {bad}; choose from {sorted(_VALID_FAMILIES)}"
            )
        return v

    @field_validator("ad_modes")
    @classmethod
    def _check_ad_modes(cls, v):
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("ad_modes cannot be an empty list; use null for all")
        bad = [m for m in v if m not in _VALID_AD_MODES]
        if bad:
            raise ValueError(f"unknown AD modes {bad}; choose from {sorted(_VALID_AD_MODES)}")
        return v

    @model_validator(mode="after")
    def _check_blocks_vs_genes(self):
        # A block partition can't request more blocks than genes.
        if self.group in ("correlation_blocks", "hierarchical_wreath") \
                and self.n_blocks > self.n_genes:
            raise ValueError(
                f"n_blocks ({self.n_blocks}) cannot exceed n_genes ({self.n_genes})"
            )
        return self


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
        description=("GEO series accession, e.g. 'GSE52778'. For a multi-platform "
                     "series, address one platform: 'GSE271850-GPL17275'."),
        examples=["GSE52778", "GSE271850-GPL17275"],
    )
    force: bool = Field(False, description="Bypass the on-disk GEO download cache.")

    @field_validator("accession")
    @classmethod
    def _check_accession(cls, v: str) -> str:
        v = v.strip().upper()
        if not _ACCESSION_RE.match(v):
            raise ValueError(
                f"{v!r} is not a GEO accession (expected e.g. 'GSE52778' or "
                "'GSE271850-GPL17275')"
            )
        return v


class MultiAnalyzeRequest(AnalyzeParams):
    """Body for ``POST /analyze/combine`` and ``POST /analyze/compare``.

    *combine* pools the accessions into one matrix (shared genes, each gene
    standardized within its own dataset to control batch effects) and runs a
    single analysis.  *compare* analyzes each accession separately and reports
    estimator agreement and top-edge overlap.
    """

    accessions: List[str] = Field(
        ...,
        min_length=2,
        max_length=8,
        description="Two or more GEO series accessions.",
        examples=[["GSE52778", "GSE147507"]],
    )
    force: bool = Field(False, description="Bypass the on-disk GEO download cache.")

    @field_validator("accessions")
    @classmethod
    def _check_accessions(cls, v):
        cleaned = []
        for a in v:
            a2 = a.strip().upper()
            if not _ACCESSION_RE.match(a2):
                raise ValueError(f"{a!r} is not a GEO accession (expected e.g. 'GSE52778')")
            cleaned.append(a2)
        if len(set(cleaned)) < 2:
            raise ValueError("provide at least two distinct accessions")
        return cleaned


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
    progress: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction complete in [0, 1] for the status bar."
    )
    phase: Optional[str] = Field(
        None, description="Short human-readable label for the current stage."
    )


class JobDetail(JobSummary):
    """Full job view including the analysis payload once it has succeeded."""

    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None


class JobList(BaseModel):
    jobs: List[JobSummary]


# ---------------------------------------------------------------------------
# GEO term search (GET /search/geo)
# ---------------------------------------------------------------------------
class GeoSearchHitModel(BaseModel):
    """One GEO series returned by the term search."""

    accession: str
    title: str = ""
    summary: str = ""
    taxon: str = ""
    n_samples: int = 0
    gds_type: str = ""
    platform: str = ""
    pub_date: str = ""
    uid: str = ""
    url: str = ""


class GeoSearchResponse(BaseModel):
    term: str
    count: int
    hits: List[GeoSearchHitModel]


# ---------------------------------------------------------------------------
# Protein-id translation (POST /translate/proteins)
# ---------------------------------------------------------------------------
class ProteinTranslateRequest(BaseModel):
    """Body for ``POST /translate/proteins`` — ids plus an optional id space."""

    ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Protein/gene identifiers: UniProt accessions, Entrez GeneIDs, or gene symbols.",
        examples=[["P04637", "7157", "TP53"]],
    )
    source: str = Field(
        "auto",
        description="Id space: auto (detect per id), uniprot, geneid, or gene.",
    )
    reviewed_only: bool = Field(
        True,
        description="Restrict GeneID/symbol lookups to reviewed (Swiss-Prot) entries.",
    )


class ProteinNameModel(BaseModel):
    query: str
    matched: bool = False
    name: str = ""
    gene: str = ""
    organism: str = ""
    accession: str = ""
    source: str = ""
    url: str = ""


class ProteinTranslateResponse(BaseModel):
    count: int
    matched: int
    results: List[ProteinNameModel]


# ---------------------------------------------------------------------------
# Gene id -> symbol translation (POST /translate/symbols)
# ---------------------------------------------------------------------------
class SymbolTranslateRequest(BaseModel):
    """Body for ``POST /translate/symbols`` — gene ids plus an organism."""

    ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Gene identifiers: Entrez GeneIDs, Ensembl ids, probe ids, aliases, or symbols.",
        examples=[["7157", "ENSG00000141510", "TP53"]],
    )
    species: str = Field(
        "human",
        description="Organism for the lookup: human (default), mouse, rat, worm (C. elegans), or an NCBI taxid.",
    )


class GeneSymbolModel(BaseModel):
    query: str
    matched: bool = False
    symbol: str = ""
    name: str = ""
    rna_type: str = ""
    type_of_gene: str = ""
    entrez: str = ""
    ensembl: str = ""
    taxid: int = 0


class SymbolTranslateResponse(BaseModel):
    count: int
    matched: int
    results: List[GeneSymbolModel]


# ---------------------------------------------------------------------------
# STRING interaction search (GET /interactions)
# ---------------------------------------------------------------------------
class InteractionModel(BaseModel):
    query: str
    partner: str
    score: float
    channels: Dict[str, float] = Field(default_factory=dict)


class InteractionsResponse(BaseModel):
    species: int
    genes: List[str]
    direct: Optional[float] = None
    partners: List[InteractionModel]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "adgencov"
    version: str
    active_jobs: int
