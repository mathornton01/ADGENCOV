"""Tests for the web dashboard (Phase D).

There is no browser or Node toolchain in this environment, so we verify the
dashboard at the boundaries we *can* drive deterministically:

1. The FastAPI app serves the SPA — ``index.html`` at ``/`` and the JS/CSS
   assets — without shadowing the API routes or the auto docs.
2. The static assets are internally consistent: the HTML declares the DOM
   hooks the JS renders into, and the JS targets the real API endpoints.
3. The analysis payload now carries what the heatmap needs — a ``covariance``
   matrix (block-orderable) matching the recommended estimator, plus its
   size-guard fallback to ``None`` for large gene sets.

Everything runs fully offline over ``TestClient`` and the committed fixtures.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import adgencov  # noqa: E402
from adgencov import analyze  # noqa: E402
from adgencov._core import recommend_estimator  # noqa: E402
from adgencov.api import create_app  # noqa: E402
from adgencov.api.jobs import JobStore  # noqa: E402
from adgencov.api.models import UploadParams  # noqa: E402
from adgencov.api.service import run_upload_analysis  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPR_FIXTURE = os.path.join(HERE, "fixtures", "expr_fixture.tsv")
STATIC_DIR = os.path.join(
    os.path.dirname(HERE), "python", "adgencov", "api", "static"
)

# DOM ids the JS renders into — the HTML must declare every one of them, or a
# render call would silently no-op in the browser.
REQUIRED_DOM_IDS = [
    "analyze-form", "accession", "file", "n_genes", "group", "top_fraction",
    "run-btn", "status", "results", "recommendation", "ranking-table",
    "blocks", "heatmap", "heatmap-legend", "network", "version",
]
# Endpoints the JS must call — keeps the client and the API contract in lockstep.
REQUIRED_ENDPOINTS = ["/analyze/geo", "/analyze/upload", "/jobs/", "/health"]


@pytest.fixture()
def client():
    store = JobStore(max_workers=1)
    app = create_app(store=store)
    with TestClient(app) as c:
        c.app.state.store = store
        yield c
    store.shutdown()


def _wait(client, job_id, timeout=60.0):
    client.app.state.store.wait(job_id, timeout=timeout)
    return client.get("/jobs/" + job_id).json()


# ---------------------------------------------------------------------------
# static assets exist on disk (so the mount has something to serve)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["index.html", "app.js", "styles.css"])
def test_static_files_present(name):
    assert os.path.isfile(os.path.join(STATIC_DIR, name)), name


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------
def test_root_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200, r.text
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "ADGENCOV" in body
    assert '/app.js' in body and '/styles.css' in body


def test_assets_served(client):
    js = client.get("/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]

    css = client.get("/styles.css")
    assert css.status_code == 200
    assert "css" in css.headers["content-type"]


def test_static_mount_does_not_shadow_api(client):
    # Explicit API routes and the auto docs must still win over the "/" mount.
    assert client.get("/health").json()["service"] == "adgencov"
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------------------
# static consistency (no browser: check the source is internally coherent)
# ---------------------------------------------------------------------------
def test_html_declares_required_dom_ids(client):
    body = client.get("/").text
    for dom_id in REQUIRED_DOM_IDS:
        assert ('id="%s"' % dom_id) in body, "missing DOM id: " + dom_id


def test_js_targets_real_endpoints(client):
    js = client.get("/app.js").text
    for ep in REQUIRED_ENDPOINTS:
        assert ep in js, "JS never references endpoint: " + ep
    # It must consume the payload keys the API produces.
    for key in ["recommended", "ranking", "edges", "covariance", "labels"]:
        assert key in js, "JS never reads payload key: " + key


# ---------------------------------------------------------------------------
# payload now carries heatmap data
# ---------------------------------------------------------------------------
def test_to_dict_includes_covariance_matching_recommendation():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 8))
    labels = [0, 0, 1, 1, 1, 2, 2, 2]
    result = analyze(X, labels)
    d = result.to_dict()

    assert d["n_genes"] == 8
    cov = d["covariance"]
    assert cov is not None
    arr = np.asarray(cov)
    assert arr.shape == (8, 8)
    # Must be exactly the recommended (rank-0) estimator's covariance.
    want = np.asarray(recommend_estimator(X, [int(v) for v in labels])[0].covariance)
    assert np.allclose(arr, want, atol=1e-12)
    # Symmetric, JSON-clean floats.
    assert np.allclose(arr, arr.T, atol=1e-9)


def test_covariance_omitted_above_size_cap(monkeypatch):
    monkeypatch.setattr(adgencov, "COVARIANCE_PAYLOAD_MAX_DIM", 4)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 8))
    labels = [0, 0, 1, 1, 1, 2, 2, 2]
    d = analyze(X, labels).to_dict()
    assert d["n_genes"] == 8
    assert d["covariance"] is None  # too big -> omitted, network still available
    assert d["edges"], "edges must survive even when the heatmap is omitted"


def test_upload_payload_has_covariance_over_http(client):
    with open(EXPR_FIXTURE, "rb") as fh:
        raw = fh.read()
    r = client.post(
        "/analyze/upload",
        files={"file": ("expr_fixture.tsv", raw, "text/tab-separated-values")},
        data={"n_genes": 6, "min_mean": 0.1, "group": "gene_family"},
    )
    assert r.status_code == 202, r.text
    detail = _wait(client, r.json()["id"])
    assert detail["state"] == "succeeded", detail.get("error")
    result = detail["result"]

    cov = result["covariance"]
    assert cov is not None
    p = result["n_genes"]
    assert np.asarray(cov).shape == (p, p)
    # Cross-check against the direct service call.
    expected = run_upload_analysis(
        raw, UploadParams(n_genes=6, min_mean=0.1, group="gene_family")
    )
    assert np.allclose(np.asarray(cov), np.asarray(expected["covariance"]), atol=1e-12)
