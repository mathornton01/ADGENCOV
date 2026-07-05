"""Offline tests for the external-database clients (:mod:`adgencov.bioquery`).

No network: a fixture-backed fake ``fetch`` dispatches on the request URL to
canned NCBI / UniProt responses captured from the live services under
``tests/fixtures/bioquery/``.  This exercises the full parse + id-matching logic
exactly as it runs against the wire, deterministically.
"""
from __future__ import annotations

import os
import urllib.parse

import pytest

from adgencov import bioquery as bq

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "bioquery")


def _load(name: str) -> bytes:
    with open(os.path.join(FIX, name), "rb") as fh:
        return fh.read()


def fake_fetch(url: str) -> bytes:
    """Route a request URL to the matching canned fixture."""
    dec = urllib.parse.unquote(url)  # queries are percent-encoded on the wire
    if "esearch.fcgi" in dec:
        return _load("esearch_gds.json")
    if "esummary.fcgi" in dec:
        return _load("esummary_gds.json")
    if "rest.uniprot.org" in dec:
        query = dec.split("query=", 1)[1] if "query=" in dec else ""
        if "accession:" in query:
            return _load("uniprot_acc.json")
        if "geneid-" in query:
            return _load("uniprot_geneid.json")
        if "gene:" in query:
            return _load("uniprot_gene.json")
    raise AssertionError(f"unexpected URL in test: {url}")


# ---------------------------------------------------------------------------
# GEO term search
# ---------------------------------------------------------------------------
def test_search_geo_returns_series_hits():
    hits = bq.search_geo("dexamethasone airway smooth muscle", fetch=fake_fetch)
    assert hits, "expected at least one series hit"
    # Every hit is a GSE series with the canonical shape.
    for h in hits:
        assert h.accession.startswith("GSE")
        assert h.url.endswith(h.accession)
        assert h.n_samples >= 0
    # The canonical example dataset is in the captured fixture set.
    accs = {h.accession for h in hits}
    assert "GSE52778" in accs
    gse = next(h for h in hits if h.accession == "GSE52778")
    assert gse.n_samples == 16
    assert "Homo sapiens" in gse.taxon
    assert gse.title


def test_search_geo_series_only_filters_non_gse():
    hits = bq.search_geo("anything", fetch=fake_fetch)
    assert all(h.accession.startswith("GSE") for h in hits)


def test_search_geo_to_dict_roundtrips():
    hits = bq.search_geo("x", fetch=fake_fetch)
    d = hits[0].to_dict()
    assert set(d) >= {"accession", "title", "n_samples", "url", "taxon"}


def test_search_geo_empty_term_raises():
    with pytest.raises(bq.BioQueryError):
        bq.search_geo("   ", fetch=fake_fetch)


def test_search_geo_no_results_returns_empty(monkeypatch):
    def empty_fetch(url: str) -> bytes:
        if "esearch.fcgi" in url:
            return b'{"esearchresult": {"idlist": []}}'
        raise AssertionError("esummary should not be called with no ids")

    assert bq.search_geo("nothing matches", fetch=empty_fetch) == []


# ---------------------------------------------------------------------------
# id classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "token,kind",
    [
        ("P04637", "uniprot"),
        ("P0DTC2", "uniprot"),
        ("A0A0B4J2F0", "uniprot"),
        ("7157", "geneid"),
        ("100", "geneid"),
        ("TP53", "gene"),
        ("Egfr", "gene"),
    ],
)
def test_classify_id(token, kind):
    assert bq.classify_id(token) == kind


# ---------------------------------------------------------------------------
# protein-id translation
# ---------------------------------------------------------------------------
def test_translate_uniprot_accessions():
    out = bq.translate_protein_ids(["P04637", "P0DTC2"], fetch=fake_fetch)
    by = {p.query: p for p in out}
    assert by["P04637"].matched and by["P04637"].name == "Cellular tumor antigen p53"
    assert by["P04637"].gene == "TP53"
    assert by["P0DTC2"].matched and by["P0DTC2"].name == "Spike glycoprotein"
    assert by["P04637"].url.endswith("/P04637/entry")


def test_translate_entrez_geneid():
    out = bq.translate_protein_ids(["7157"], fetch=fake_fetch)
    assert len(out) == 1
    p = out[0]
    assert p.matched and p.source == "geneid"
    assert p.gene == "TP53"
    assert p.accession == "P04637"


def test_translate_gene_symbol():
    out = bq.translate_protein_ids(["TP53"], source="gene", fetch=fake_fetch)
    p = out[0]
    assert p.matched and p.source == "gene"
    assert "p53" in p.name.lower()


def test_translate_mixed_auto_preserves_order_and_unmatched():
    ids = ["P04637", "7157", "TP53", "ZZZNOTREAL"]
    out = bq.translate_protein_ids(ids, fetch=fake_fetch)
    assert [p.query for p in out] == ids  # order preserved
    assert out[-1].matched is False       # unknown symbol → unmatched placeholder
    assert all(p.matched for p in out[:3])


def test_translate_dedups_repeated_ids():
    out = bq.translate_protein_ids(["P04637", "P04637", "P0DTC2"], fetch=fake_fetch)
    assert [p.query for p in out] == ["P04637", "P0DTC2"]


def test_translate_empty_list_returns_empty():
    assert bq.translate_protein_ids([], fetch=fake_fetch) == []
    assert bq.translate_protein_ids(["", "   "], fetch=fake_fetch) == []


def test_translate_unknown_source_raises():
    with pytest.raises(bq.BioQueryError):
        bq.translate_protein_ids(["P04637"], source="bogus", fetch=fake_fetch)


def test_translate_to_dict_shape():
    p = bq.translate_protein_ids(["P04637"], fetch=fake_fetch)[0]
    d = p.to_dict()
    assert set(d) == {
        "query", "matched", "name", "gene", "organism", "accession", "source", "url"
    }
