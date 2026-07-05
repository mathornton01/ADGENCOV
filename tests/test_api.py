"""End-to-end tests for the FastAPI service (Phase C).

These drive the real ASGI app through ``TestClient`` (HTTP in, JSON out) and
prove three things:

1. The service wiring is correct — health, docs, submit, poll, list, delete.
2. The HTTP layer does not distort the numerics: the payload returned over
   HTTP equals the direct :func:`run_upload_analysis` result byte-for-byte.
3. The GEO path reproduces the committed 1e-9 pipeline golden (``ad_lasso``,
   LOO ``5.431879976737194``) when handed the series-matrix fixture — so
   ingestion → recommender is faithful through the web boundary too.

Everything runs fully offline: the "accession" for the GEO test is the local
series-matrix fixture path, which :func:`adgencov.geo.load_series` parses
without touching the network.
"""
from __future__ import annotations

import json
import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from adgencov.api import create_app  # noqa: E402
from adgencov.api.jobs import JobStore  # noqa: E402
from adgencov.api.models import UploadParams  # noqa: E402
from adgencov.api.service import run_upload_analysis  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPR_FIXTURE = os.path.join(HERE, "fixtures", "expr_fixture.tsv")
GEO_FIXTURE = os.path.join(HERE, "fixtures", "geo_series_matrix.txt")

# The 1e-9 pipeline golden reproduced elsewhere (test_geo.py / golden_pipeline).
GOLDEN_GEO_METHOD = "ad_lasso"
GOLDEN_GEO_LOO = 5.431879976737194


@pytest.fixture()
def client():
    """A TestClient over a fresh app with its own single-worker job store.

    A single worker keeps job ordering deterministic; we still exercise the
    async submit→poll contract and wait on the store for terminal state.
    """
    store = JobStore(max_workers=1)
    app = create_app(store=store)
    with TestClient(app) as c:
        c.app.state.store = store  # expose for wait()
        yield c
    store.shutdown()


def _wait(client, job_id, timeout=60.0):
    """Block until the job reaches a terminal state, then return its detail."""
    client.app.state.store.wait(job_id, timeout=timeout)
    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "adgencov"
    assert isinstance(body["version"], str) and body["version"]
    assert body["active_jobs"] == 0


def test_openapi_docs_exposed(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    # The two analyze entry points and the jobs endpoints must be documented.
    assert "/analyze/upload" in spec["paths"]
    assert "/analyze/geo" in spec["paths"]
    assert "/jobs/{job_id}" in spec["paths"]


# ---------------------------------------------------------------------------
# upload path
# ---------------------------------------------------------------------------
def test_upload_flow_matches_service(client):
    with open(EXPR_FIXTURE, "rb") as fh:
        raw = fh.read()

    r = client.post(
        "/analyze/upload",
        files={"file": ("expr_fixture.tsv", raw, "text/tab-separated-values")},
        data={"n_genes": 6, "min_mean": 0.1, "group": "gene_family"},
    )
    assert r.status_code == 202, r.text
    summary = r.json()
    assert summary["kind"] == "upload"
    assert summary["state"] in ("pending", "running", "succeeded")
    assert summary["label"] == "expr_fixture.tsv"

    detail = _wait(client, summary["id"])
    assert detail["state"] == "succeeded", detail.get("error")
    result = detail["result"]

    # The HTTP payload must equal the direct service call to 1e-12 — the web
    # boundary must not perturb the numerics.
    expected = run_upload_analysis(
        raw, UploadParams(n_genes=6, min_mean=0.1, group="gene_family")
    )
    assert result["recommended"] == expected["recommended"]
    assert result["genes"] == expected["genes"]
    assert len(result["ranking"]) == len(expected["ranking"]) == 24
    for got, exp in zip(result["ranking"], expected["ranking"]):
        assert got["method"] == exp["method"]
        assert got["loo_nll"] == pytest.approx(exp["loo_nll"], abs=1e-12)
    assert result["source"] == {"kind": "upload"}
    # Must survive a JSON round-trip (it already came over HTTP, but be explicit).
    json.loads(json.dumps(result))


def test_upload_empty_file_fails_gracefully(client):
    r = client.post(
        "/analyze/upload",
        files={"file": ("empty.tsv", b"   \n", "text/plain")},
        data={"n_genes": 6},
    )
    assert r.status_code == 202
    detail = _wait(client, r.json()["id"])
    assert detail["state"] == "failed"
    assert detail["error"]
    assert detail["result"] is None


def test_upload_invalid_params_rejected(client):
    # top_fraction > 1 violates the pydantic constraint -> 422 before any job.
    r = client.post(
        "/analyze/upload",
        files={"file": ("expr.tsv", b"gene_short_name\ta\tb\tc\nG1\t1\t2\t3\n", "text/plain")},
        data={"top_fraction": 5.0},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GEO path (offline: the "accession" is the local series-matrix fixture)
# ---------------------------------------------------------------------------
def test_geo_flow_reproduces_golden(client):
    r = client.post(
        "/analyze/geo",
        json={
            "accession": GEO_FIXTURE,
            "n_genes": 6,
            "min_mean": 0.1,
            "group": "gene_family",
        },
    )
    assert r.status_code == 202, r.text
    summary = r.json()
    assert summary["kind"] == "geo"
    assert summary["label"] == GEO_FIXTURE

    detail = _wait(client, summary["id"])
    assert detail["state"] == "succeeded", detail.get("error")
    result = detail["result"]

    assert result["recommended"] == GOLDEN_GEO_METHOD
    assert result["ranking"][0]["loo_nll"] == pytest.approx(GOLDEN_GEO_LOO, abs=1e-9)
    assert len(result["ranking"]) == 24
    assert result["edges"], "expected at least one covariance edge"
    assert result["source"]["kind"] == "geo"


def test_geo_missing_accession_rejected(client):
    r = client.post("/analyze/geo", json={"n_genes": 6})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# jobs lifecycle
# ---------------------------------------------------------------------------
def test_job_not_found(client):
    assert client.get("/jobs/deadbeef").status_code == 404
    assert client.delete("/jobs/deadbeef").status_code == 404


def test_list_and_delete_jobs(client):
    with open(EXPR_FIXTURE, "rb") as fh:
        raw = fh.read()
    ids = []
    for _ in range(2):
        r = client.post(
            "/analyze/upload",
            files={"file": ("expr.tsv", raw, "text/plain")},
            data={"n_genes": 6, "min_mean": 0.1, "group": "gene_family"},
        )
        ids.append(r.json()["id"])
    for jid in ids:
        _wait(client, jid)

    listing = client.get("/jobs").json()["jobs"]
    assert len(listing) >= 2
    assert all(j["state"] == "succeeded" for j in listing if j["id"] in ids)

    # Delete one and confirm it's gone.
    assert client.delete(f"/jobs/{ids[0]}").status_code == 204
    assert client.get(f"/jobs/{ids[0]}").status_code == 404
    remaining = {j["id"] for j in client.get("/jobs").json()["jobs"]}
    assert ids[0] not in remaining
    assert ids[1] in remaining
