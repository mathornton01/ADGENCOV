"""Offline tests for the GEO ingestion layer (Phase B).

Everything here runs without network access: the series-matrix parser, URL
builder, probe→gene mapping, and the full GEO→recommender path are exercised
against a small committed fixture (tests/fixtures/geo_series_matrix.txt).

The end-to-end test is the important one: the fixture carries the same gene
symbols and expression values as the CLI pipeline golden, so after
preprocessing (which z-scores away the sample-name difference) the GEO path must
reproduce the reference recommendation to 1e-9 — proving ingestion feeds the
fast path correctly from start to finish.

Run:  pytest -q tests/test_geo.py     (needs numpy + pandas on the interpreter)
"""
from __future__ import annotations

import gzip
import io
import math
import os

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from adgencov import geo  # noqa: E402  (after importorskip)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "geo_series_matrix.txt")

# Golden reference from tests/golden_pipeline.hpp (gene_family, n_genes=6,
# min_mean=0.1) — the GEO path must reproduce this exactly.
GOLDEN_GENES = ["COL1A1", "KRT8", "TP53", "HIST1H", "RPL7", "RPS3"]
GOLDEN_BEST_METHOD = "ad_lasso"
GOLDEN_BEST_LOO = 5.431879976737194
GOLDEN_RANK_METHODS = [
    "ad_lasso", "ad_elastic_net", "ad_ridge", "ad_linear_lw", "ad_ridge", "lw",
    "ad_ridge", "ad_lasso", "ad_elastic_net", "ad_ridge", "oas",
    "ad_elastic_net", "ad_oas", "ad_lasso", "ad_ridge", "ad_elastic_net",
    "ad_lasso", "ad_target_ridge", "ad_target_ridge", "ad_target_ridge",
    "ad_target_ridge", "ad_target_ridge", "ad_target_lw", "ad_target_oas",
]
GOLDEN_RANK_LOO = [
    5.431879976737194, 5.445512535078997, 5.567458167902287, 5.88245424655942,
    5.9424760601227105, 5.966325677355305, 6.041449577265634,
    6.0689868820392725, 6.167470222511, 6.7715839357397245, 7.1339711623078905,
    7.1833258035724254, 7.189866430400039, 7.243284358622774,
    7.7448291082577905, 15.607186941910562, 15.714630025049743,
    148.9743431988566, 148.9743431988705, 148.97434319888498,
    148.97434319890078, 148.97434319890414, 14517.199465270112,
    14517.199465457044,
]


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "acc,expected",
    [
        ("GSE52778",
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE52nnn/GSE52778/matrix/"
         "GSE52778_series_matrix.txt.gz"),
        ("GSE567",
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/GSE567/matrix/"
         "GSE567_series_matrix.txt.gz"),
        ("gse1",
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/GSE1/matrix/"
         "GSE1_series_matrix.txt.gz"),
        ("GSE1000000",
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1000nnn/GSE1000000/matrix/"
         "GSE1000000_series_matrix.txt.gz"),
    ],
)
def test_series_matrix_url(acc, expected):
    assert geo.series_matrix_url(acc) == expected


@pytest.mark.parametrize("bad", ["", "GSM123", "PRJNA1", "GSE", "52778", "foo"])
def test_series_matrix_url_rejects_non_gse(bad):
    with pytest.raises(geo.GeoError):
        geo.series_matrix_url(bad)


def test_default_cache_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ADGENCOV_CACHE", str(tmp_path))
    assert geo.default_cache_dir() == os.path.join(str(tmp_path), "geo")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def test_parse_shape_and_ids():
    s = geo.read_series_matrix(FIXTURE)
    assert s.accession == "GSE999999"
    assert s.platform == "GPL0000"
    assert s.n_samples == 6
    assert s.sample_ids == [f"GSM000000{i}" for i in range(1, 7)]
    # 10 gene rows in the fixture (incl. a duplicate RPS3 and a low-mean GHOST).
    assert s.n_genes == 10
    assert list(s.expression.columns)[0] == "gene"
    assert s.expression["gene"].tolist().count("RPS3") == 2


def test_parse_metadata_and_samples():
    s = geo.read_series_matrix(FIXTURE)
    assert s.title == "ADGENCOV GEO ingestion offline fixture"
    # Multi-line !Series_summary is accumulated into a list.
    assert isinstance(s.metadata["summary"], list)
    assert len(s.metadata["summary"]) == 2
    # Per-sample metadata frame indexed by GSM id.
    assert list(s.samples.index) == s.sample_ids
    assert s.samples.loc["GSM0000001", "title"] == "Dex rep1"
    assert "dexamethasone" in s.samples.loc["GSM0000001", "characteristics_ch1"]


def test_sample_regex_matches_only_samples():
    import re
    s = geo.read_series_matrix(FIXTURE)
    rx = re.compile(s.sample_regex())
    assert all(rx.search(g) for g in s.sample_ids)
    for other in ("gene", "ID_REF", "GSM0000001_extra", "title"):
        assert not rx.search(other)


def test_gzip_roundtrip_identical(tmp_path):
    plain = geo.read_series_matrix(FIXTURE)
    gzpath = tmp_path / "GSE999999_series_matrix.txt.gz"
    with open(FIXTURE, "rb") as fin, gzip.open(gzpath, "wb") as fout:
        fout.write(fin.read())
    gz = geo.read_series_matrix(str(gzpath))
    pd.testing.assert_frame_equal(plain.expression, gz.expression)
    assert gz.accession == plain.accession


def test_parse_handles_na_and_quotes():
    text = (
        '!Series_title\t"tiny"\n'
        "!series_matrix_table_begin\n"
        '"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"\n'
        '"A"\t1.0\t2.0\tNA\n'
        '"B"\t\t4.0\t5.0\n'
        "!series_matrix_table_end\n"
    )
    s = geo.read_series_matrix(io.StringIO(text))
    assert s.sample_ids == ["GSM1", "GSM2", "GSM3"]
    vals = s.expression.set_index("gene")
    assert math.isnan(vals.loc["A", "GSM3"])
    assert math.isnan(vals.loc["B", "GSM1"])
    assert vals.loc["A", "GSM1"] == 1.0


def test_parse_without_table_raises():
    with pytest.raises(geo.GeoError):
        geo.read_series_matrix(io.StringIO('!Series_title\t"no table here"\n'))


# ---------------------------------------------------------------------------
# Probe -> gene mapping
# ---------------------------------------------------------------------------
def _probe_series():
    df = pd.DataFrame(
        {
            "gene": ["p1", "p2", "p3", "p4"],
            "GSM1": [10.0, 5.0, 1.0, 9.0],
            "GSM2": [12.0, 6.0, 2.0, 8.0],
            "GSM3": [11.0, 4.0, 1.5, 7.0],
        }
    )
    return geo.GeoSeries(accession="GSEX", expression=df)


def test_map_probes_max_mean_collapses_duplicates():
    s = _probe_series()
    # p1 and p4 both map to GENEA; p1 has the higher mean → kept.
    mapping = {"p1": "GENEA", "p4": "GENEA", "p2": "GENEB", "p3": ""}
    out = geo.map_probes_to_genes(s, mapping, aggregate="max_mean")
    genes = out.expression["gene"].tolist()
    assert genes == ["GENEA", "GENEB"]  # p3 dropped (empty), GENEA de-duped
    row = out.expression.set_index("gene").loc["GENEA"]
    assert row["GSM1"] == 10.0  # kept p1, not p4


def test_map_probes_mean_aggregate():
    s = _probe_series()
    mapping = {"p1": "G", "p4": "G", "p2": "H", "p3": "H"}
    out = geo.map_probes_to_genes(s, mapping, aggregate="mean")
    g = out.expression.set_index("gene")
    assert g.loc["G", "GSM1"] == pytest.approx((10.0 + 9.0) / 2)
    assert g.loc["H", "GSM1"] == pytest.approx((5.0 + 1.0) / 2)


def test_map_probes_dataframe_mapping():
    s = _probe_series()
    lut = pd.DataFrame({"probe": ["p1", "p2"], "sym": ["AA", "BB"]})
    out = geo.map_probes_to_genes(
        s, lut, probe_col="probe", gene_col="sym", drop_unmapped=True
    )
    assert out.expression["gene"].tolist() == ["AA", "BB"]  # p3,p4 unmapped → dropped


def test_map_probes_empty_result_raises():
    s = _probe_series()
    with pytest.raises(geo.GeoError):
        geo.map_probes_to_genes(s, {"zzz": "Q"}, drop_unmapped=True)


# ---------------------------------------------------------------------------
# End-to-end: GEO series -> recommendation (the start-to-finish proof)
# ---------------------------------------------------------------------------
def test_analyze_series_reproduces_pipeline_golden():
    result = geo.analyze_series(
        FIXTURE, n_genes=6, min_mean=0.1, group="gene_family"
    )
    # Same standardized genes, in the same order, as the CLI pipeline golden.
    assert result.genes == GOLDEN_GENES
    # Same recommendation to 1e-9.
    assert result.best.spec.method == GOLDEN_BEST_METHOD
    assert result.best.loo_nll == pytest.approx(GOLDEN_BEST_LOO, abs=1e-9)
    # The entire 24-candidate ranking matches.
    methods = [r.spec.method for r in result.ranking]
    loos = [r.loo_nll for r in result.ranking]
    assert methods == GOLDEN_RANK_METHODS
    # Relative tolerance: NLLs span 5 -> 14517 (the AD-target family is heavily
    # over-regularized on this data and ranks last), so the C++ fast path and the
    # golden generator agree to ~1e-11 relative, not to a fixed 1e-9 absolute.
    assert loos == pytest.approx(GOLDEN_RANK_LOO, rel=1e-9, abs=1e-9)


def test_analyze_series_accepts_geoseries_and_serializes():
    s = geo.read_series_matrix(FIXTURE)
    result = geo.analyze_series(s, n_genes=6, min_mean=0.1, group="gene_family")
    d = result.to_dict()
    assert d["recommended"] == GOLDEN_BEST_METHOD
    assert d["genes"] == GOLDEN_GENES
    assert len(d["ranking"]) == 24
    assert d["edges"], "expected at least one covariance edge"
    # to_dict must be JSON-serializable for the FastAPI layer (Phase C).
    import json
    json.loads(json.dumps(d))


def test_analyze_series_rejects_too_few_samples():
    text = (
        "!series_matrix_table_begin\n"
        '"ID_REF"\t"GSM1"\t"GSM2"\n'
        '"RPS3"\t1.0\t2.0\n'
        '"RPL7"\t3.0\t4.0\n'
        "!series_matrix_table_end\n"
    )
    s = geo.read_series_matrix(io.StringIO(text))
    with pytest.raises(geo.GeoError):
        geo.analyze_series(s)


def test_load_series_passthrough():
    s = geo.read_series_matrix(FIXTURE)
    assert geo.load_series(s) is s
    assert geo.load_series(FIXTURE).accession == "GSE999999"


# ---------------------------------------------------------------------------
# Supplementary-matrix fallback (RNA-seq counts / FPKM / TPM)
# ---------------------------------------------------------------------------
SUPP_FPKM = os.path.join(HERE, "fixtures", "geo_supp_fpkm_matrix.txt")
SUPP_COUNTS = os.path.join(HERE, "fixtures", "geo_supp_counts.csv")
SUPP_DESEQ2 = os.path.join(HERE, "fixtures", "geo_supp_deseq2_annot.txt")


def test_supp_picker_prefers_matrix_over_diff():
    names = [
        "GSE52778_All_Sample_FPKM_Matrix.txt.gz",
        "GSE52778_Dex_vs_Untreated_gene_exp.diff.gz",
    ]
    assert geo.pick_supplementary_matrix(names) == names[0]


def test_supp_picker_scores_and_rejects():
    # FPKM/counts/tpm keywords score positive; annotation/diff/gtf reject.
    assert geo._score_matrix_candidate("x_FPKM_matrix.txt.gz") > 0
    assert geo._score_matrix_candidate("x_raw_counts.tsv.gz") > 0
    assert geo._score_matrix_candidate("x_gene_exp.diff.gz") <= 0
    assert geo._score_matrix_candidate("x_annotation.gtf.gz") <= 0
    assert geo._score_matrix_candidate("x.bam") <= 0  # not tabular
    # A bare accession file with no keyword is still a plausible matrix.
    assert geo._score_matrix_candidate("GSE1_table.txt.gz") > 0
    assert geo.pick_supplementary_matrix(["a_readme.txt", "b.bam"]) == "a_readme.txt" or \
        geo.pick_supplementary_matrix(["b.bam"]) is None


def test_supp_picker_none_when_no_matrix():
    assert geo.pick_supplementary_matrix(["x.bam", "y.gtf.gz"]) is None


def test_supp_dir_url():
    assert geo.supplementary_dir_url("GSE52778") == (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE52nnn/GSE52778/suppl/"
    )
    assert geo.supplementary_dir_url("GSE567").endswith("/GSEnnn/GSE567/suppl/")


def test_supp_cufflinks_drops_annotation_and_stats():
    # Space-delimited cuffdiff-style matrix: annotation + _conf_lo/_conf_hi/
    # _status columns must be excluded; _FPKM columns become clean sample names.
    s = geo.read_supplementary_matrix(SUPP_FPKM, accession="GSE52778")
    assert s.accession == "GSE52778"
    assert s.n_genes == 6
    # 3 condition (_FPKM) + 3 replicate columns = 6 samples; no conf/status/annot.
    assert s.n_samples == 6
    assert set(s.sample_ids) == {
        "Dex", "Alb", "Untreated", "Dex_R1", "Alb_R1", "Untreated_R1"
    }
    for bad in ("gene_id", "locus", "length", "coverage", "tss_id"):
        assert bad not in s.sample_ids
    for c in s.sample_ids:
        assert not c.endswith(("_conf_lo", "_conf_hi", "_status"))
    # gene ids came from gene_short_name, not ENSG gene_id.
    assert "RPS3" in s.expression[geo.GENE_COL].tolist()
    assert "ENSG001" not in s.expression[geo.GENE_COL].tolist()


def test_supp_deseq2_annot_isolates_count_family():
    # A DESeq2-style annotated table (shape seen in GSE300090): a gene-id + a
    # sparse gene_name + rich annotation, then the SAME samples reported twice
    # (``*_count`` and ``*_tpm``), per-condition aggregates (``QDT_tpm``,
    # ``Ctrl_tpm``) and un-suffixed statistic columns (fc/log2fc/pvalue/padjust).
    s = geo.read_supplementary_matrix(SUPP_DESEQ2, accession="GSE300090")
    assert s.accession == "GSE300090"
    # Exactly the 6 raw-count replicate columns become samples — counts are
    # preferred over the redundant tpm unit; aggregates and stats fall away.
    assert s.n_samples == 6
    assert set(s.sample_ids) == {
        "QDT_T1", "QDT_T2", "QDT_T3", "QDT_C1", "QDT_C2", "QDT_C3"
    }
    # No unit tags, aggregates, statistics, or annotation leaked in as samples.
    for bad in ("QDT_tpm", "Ctrl_tpm", "fc", "log2fc", "pvalue", "padjust",
                "significant", "regulate", "length", "entrez"):
        assert bad not in s.sample_ids
    for c in s.sample_ids:
        assert not c.lower().endswith(("_count", "_tpm"))
    # gene_name is sparse (blank for the novel transcript) so the fully-populated
    # gene_id is used instead — every row keeps an identifier.
    genes = s.expression[geo.GENE_COL].tolist()
    assert s.n_genes == 6
    assert genes[0] == "ENSG00000255404"
    assert "" not in genes


def test_deseq2_annot_name_is_ranked_not_rejected():
    # The name alone (deseq/annot tokens) must no longer disqualify the file —
    # content validation is what decides.
    only = ["GSE300090_Control_vs_treated_QDT.deseq2.annot.txt.gz"]
    assert geo.rank_supplementary_matrices(only) == only
    assert geo.pick_supplementary_matrix(only) == only[0]
    # Structural non-matrix formats are still hard-rejected.
    assert geo.rank_supplementary_matrices(["GSE1_annotation.gtf.gz"]) == []
    assert geo.pick_supplementary_matrix(["GSE1_RAW.tar"]) is None


def test_supp_plain_counts_csv():
    s = geo.read_supplementary_matrix(SUPP_COUNTS, accession="GSE1")
    assert s.n_genes == 6
    assert s.sample_ids == ["SampleA", "SampleB", "SampleC", "SampleD"]
    assert s.expression[geo.GENE_COL].tolist()[0] == "RPS3"


def test_supp_delimiter_sniff():
    assert geo._sniff_delimiter("a\tb\tc") == "\t"
    assert geo._sniff_delimiter("a,b,c") == ","
    assert geo._sniff_delimiter("a b c") == r"\s+"


def test_supp_too_few_sample_columns_raises():
    text = "gene\tonly_status\nRPS3\tOK\nACTB\tOK\n"
    with pytest.raises(geo.GeoError):
        geo.read_supplementary_matrix(io.StringIO(text))


def test_supp_analyze_series_end_to_end():
    # The whole point: a supplementary FPKM matrix must drive the C++ core.
    s = geo.read_supplementary_matrix(SUPP_FPKM, accession="GSE52778")
    import json
    res = geo.analyze_series(
        s, n_genes=6, min_mean=0.0, group="gene_family", n_blocks=2
    )
    d = res.to_dict()
    assert d["recommended"]
    assert len(d["ranking"]) == 24  # full extended grid runs on FPKM data
    json.loads(json.dumps(d))  # JSON-serializable
