"""Offline tests for gene→symbol translation (mygene) and STRING interactions.

No network: injected ``fetch_post`` callables return canned mygene / STRING
payloads, so the full parse + shaping logic runs deterministically.  The HTTP
tests monkeypatch the module-level ``default_post_fetch`` and point the on-disk
cache at a tmp dir so nothing touches the wire or the developer's home cache.
"""
from __future__ import annotations

import json

import pytest

from adgencov import bioquery as bq
from adgencov import stringdb as sd


# ---------------------------------------------------------------------------
# fixtures: canned upstream responses
# ---------------------------------------------------------------------------
_MYGENE_ROWS = [
    {"query": "7157", "symbol": "TP53", "name": "tumor protein p53",
     "type_of_gene": "protein-coding", "taxid": 9606, "entrezgene": 7157,
     "ensembl": {"gene": "ENSG00000141510"}},
    {"query": "MIR21", "symbol": "MIR21", "name": "microRNA 21",
     "type_of_gene": "ncRNA", "taxid": 9606},
    {"query": "SNORD44", "symbol": "SNORD44", "name": "small nucleolar RNA, C/D box 44",
     "type_of_gene": "snoRNA", "taxid": 9606},
    {"query": "nope", "notfound": True},
]
_STRING_ROWS = [
    {"preferredName_A": "TP53", "preferredName_B": "MDM2", "score": 0.999,
     "escore": 0.9, "dscore": 0.8, "tscore": 0.7},
    {"preferredName_A": "TP53", "preferredName_B": "EP300", "score": 0.951},
    {"preferredName_A": "MDM2", "preferredName_B": "TP53", "score": 0.999},
]


def _mygene_fetch(url, fields):
    assert "mygene" in url
    assert fields["q"]  # ids were forwarded
    return json.dumps(_MYGENE_ROWS).encode()


def _string_fetch(url, fields):
    assert "string" in url
    assert fields["identifiers"]
    return json.dumps(_STRING_ROWS).encode()


# ---------------------------------------------------------------------------
# gene → symbol (mygene) unit behaviour
# ---------------------------------------------------------------------------
def test_translate_gene_symbols_resolves_and_shapes():
    out = bq.translate_gene_symbols(
        ["7157", "MIR21", "SNORD44", "nope"], fetch_post=_mygene_fetch, use_cache=False
    )
    by_q = {g.query: g for g in out}
    assert by_q["7157"].symbol == "TP53"
    assert by_q["7157"].rna_type == "protein_coding"
    assert by_q["7157"].ensembl == "ENSG00000141510"
    # Symbol-prefix heuristic overrides a generic ncRNA biotype.
    assert by_q["MIR21"].rna_type == "miRNA"
    assert by_q["SNORD44"].rna_type == "snoRNA"
    # Unresolved id comes back as an unmatched placeholder in original order.
    assert by_q["nope"].matched is False
    assert [g.query for g in out] == ["7157", "MIR21", "SNORD44", "nope"]


def test_translate_gene_symbols_dedups_and_preserves_order():
    seen = {}

    def fetch(url, fields):
        seen["q"] = fields["q"]
        return json.dumps([{"query": "TP53", "symbol": "TP53",
                            "type_of_gene": "protein-coding"}]).encode()

    out = bq.translate_gene_symbols(["TP53", "TP53", ""], fetch_post=fetch, use_cache=False)
    assert len(out) == 1 and out[0].symbol == "TP53"
    assert seen["q"] == "TP53"  # duplicate + blank collapsed to a single id


def test_rna_type_of_heuristics():
    assert bq.rna_type_of("MIR155", "") == "miRNA"
    assert bq.rna_type_of("SNORD44", "") == "snoRNA"
    assert bq.rna_type_of("TP53", "protein-coding") == "protein_coding"
    assert bq.rna_type_of("XIST", "ncRNA") == "ncRNA"


def test_symbol_cache_avoids_second_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("ADGENCOV_CACHE_DIR", str(tmp_path))
    # Reset the in-process cache mirror so the tmp dir is authoritative.
    from adgencov import biocache
    biocache._MEM.clear()

    calls = {"n": 0}

    def counting_fetch(url, fields):
        calls["n"] += 1
        return json.dumps([{"query": "7157", "symbol": "TP53",
                            "type_of_gene": "protein-coding", "taxid": 9606}]).encode()

    a = bq.translate_gene_symbols(["7157"], fetch_post=counting_fetch)
    b = bq.translate_gene_symbols(["7157"], fetch_post=counting_fetch)
    assert a[0].symbol == b[0].symbol == "TP53"
    assert calls["n"] == 1  # second call served from cache


# ---------------------------------------------------------------------------
# STRING interactions unit behaviour
# ---------------------------------------------------------------------------
def test_interactions_parses_partners_and_direct_edge():
    data = sd.interactions(["TP53", "MDM2"], fetch_post=_string_fetch, use_cache=False)
    assert data["species"] == 9606
    assert data["direct"] == pytest.approx(0.999)  # TP53–MDM2 both queried
    partners = data["partners"]
    assert partners[0]["score"] >= partners[-1]["score"]  # sorted strongest-first
    tp53_ep300 = next(p for p in partners if p["partner"] == "EP300")
    assert tp53_ep300["query"] == "TP53"


def test_interactions_species_alias_and_no_direct_for_single():
    data = sd.interactions(["TP53"], species="mouse", fetch_post=_string_fetch, use_cache=False)
    assert data["species"] == 10090
    assert data["direct"] is None  # a single gene has no queried-pair edge


def test_interactions_rejects_unknown_species():
    with pytest.raises(sd.StringError):
        sd.interactions(["TP53"], species="banana", fetch_post=_string_fetch, use_cache=False)


# ---------------------------------------------------------------------------
# HTTP endpoints (monkeypatched fetchers, tmp cache)
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from adgencov.api import create_app
    from adgencov.api.jobs import JobStore
    from adgencov import biocache

    monkeypatch.setenv("ADGENCOV_CACHE_DIR", str(tmp_path))
    biocache._MEM.clear()
    monkeypatch.setattr(bq, "default_post_fetch", lambda *a, **k: _mygene_fetch)
    monkeypatch.setattr(sd, "default_post_fetch", lambda *a, **k: _string_fetch)

    store = JobStore(max_workers=1)
    app = create_app(store=store)
    with TestClient(app) as c:
        yield c
    store.shutdown()


def test_translate_symbols_endpoint(client):
    r = client.post("/translate/symbols",
                    json={"ids": ["7157", "MIR21", "nope"], "species": "human"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 3
    assert body["matched"] == 2  # 7157→TP53 and MIR21 resolve; "nope" is notfound
    syms = {row["query"]: row for row in body["results"]}
    assert syms["7157"]["symbol"] == "TP53"
    assert syms["MIR21"]["rna_type"] == "miRNA"


def test_interactions_endpoint(client):
    r = client.get("/interactions", params={"genes": "TP53,MDM2", "species": "human"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["species"] == 9606
    assert body["direct"] == pytest.approx(0.999)
    assert any(p["partner"] == "EP300" for p in body["partners"])


def test_interactions_rejects_too_many_genes(client):
    r = client.get("/interactions", params={"genes": "A,B,C"})
    assert r.status_code == 422
