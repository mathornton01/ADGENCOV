// End-to-end pipeline tests: load -> preprocess -> group -> recommend, checked
// against golden values produced from the prototype by gen_golden_pipeline.py.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <string>
#include <vector>

#include "adgencov/groups.hpp"
#include "adgencov/io.hpp"
#include "adgencov/preprocess.hpp"
#include "adgencov/select.hpp"

#include "golden_pipeline.hpp"

#ifndef ADGENCOV_TEST_DIR
#define ADGENCOV_TEST_DIR "."
#endif

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace adgencov;

namespace {
Dataset load_fixture() {
  const std::string fixture = std::string(ADGENCOV_TEST_DIR) + "/fixtures/expr_fixture.tsv";
  ExpressionData raw = load_expression_matrix(fixture, "_LL[0-9]+", "gene_short_name");
  return preprocess(raw, /*n_genes=*/6, /*min_mean=*/0.1, /*log_transform=*/true);
}
}  // namespace

TEST_CASE("gene_family_label matches the prototype heuristic", "[groups]") {
  REQUIRE(gene_family_label("RPS3") == "RPS");
  REQUIRE(gene_family_label("RPL7") == "RPL");
  REQUIRE(gene_family_label("COL1A1") == "COL");
  REQUIRE(gene_family_label("mir21") == "MIR");
  REQUIRE(gene_family_label("TP53") == "TP");    // leading alpha run
  REQUIRE(gene_family_label("HIST1H") == "HIST");
  REQUIRE(gene_family_label("12ORF") == "OTHER");  // no leading alpha
}

TEST_CASE("factorize assigns dense codes in first-appearance order", "[groups]") {
  std::vector<std::string> in = {"b", "a", "b", "c", "a"};
  std::vector<int> got = factorize(in);
  REQUIRE(got == std::vector<int>({0, 1, 0, 2, 1}));
}

TEST_CASE("preprocess reproduces the prototype's standardized matrix", "[preprocess]") {
  Dataset ds = load_fixture();
  REQUIRE(ds.X.rows() == golden_pipeline::kNSamples);
  REQUIRE(ds.X.cols() == golden_pipeline::kNGenes);
  REQUIRE(ds.genes == golden_pipeline::kGenes);  // selection + ordering

  for (int i = 0; i < ds.X.rows(); ++i)
    for (int j = 0; j < ds.X.cols(); ++j) {
      double gold = golden_pipeline::kX[i * ds.X.cols() + j];
      REQUIRE_THAT(ds.X(i, j), WithinAbs(gold, 1e-9));
    }
}

TEST_CASE("build_group_labels(gene_family) matches the prototype", "[groups]") {
  Dataset ds = load_fixture();
  std::vector<std::string> labels = build_group_labels(ds, "gene_family");
  REQUIRE(labels == golden_pipeline::kLabels);
}

TEST_CASE("build_group_labels errors when required tables are missing", "[groups]") {
  Dataset ds = load_fixture();
  REQUIRE_THROWS(build_group_labels(ds, "chromosome"));
  REQUIRE_THROWS(build_group_labels(ds, "custom_group_map"));
  REQUIRE_THROWS(build_group_labels(ds, "correlation_blocks"));  // deferred
  REQUIRE_THROWS(build_group_labels(ds, "nonsense"));
}

TEST_CASE("group map and chromosome tables map genes correctly", "[groups]") {
  Dataset ds = load_fixture();  // genes: COL1A1 KRT8 TP53 HIST1H RPL7 RPS3

  Table gmap;
  gmap.headers = {"gene", "group"};
  gmap.rows = {{"COL1A1", "matrix"}, {"KRT8", "matrix"}, {"RPS3", "ribosome"}};
  std::vector<std::string> gl = build_group_labels(ds, "custom_group_map", nullptr, &gmap);
  REQUIRE(gl[0] == "matrix");     // COL1A1
  REQUIRE(gl[1] == "matrix");     // KRT8
  REQUIRE(gl[2] == "unmapped");   // TP53 absent
  REQUIRE(gl[5] == "ribosome");   // RPS3

  Table ann;
  ann.headers = {"gene", "chromosome"};
  ann.rows = {{"COL1A1", "chr17"}, {"TP53", "chr17"}};
  std::vector<std::string> cl = build_group_labels(ds, "chromosome", &ann, nullptr);
  REQUIRE(cl[0] == "chr17");        // COL1A1
  REQUIRE(cl[2] == "chr17");        // TP53
  REQUIRE(cl[1] == "chr_unknown");  // KRT8 absent
}

TEST_CASE("recommend_estimator reproduces the prototype ranking", "[pipeline]") {
  Dataset ds = load_fixture();
  std::vector<std::string> label_names = build_group_labels(ds, "gene_family");
  std::vector<int> labels = factorize(label_names);

  std::vector<EstimatorResult> results = recommend_estimator(ds.X, labels);
  REQUIRE(results.size() == golden_pipeline::kRankMethods.size());

  // Best method + LOO NLL match the reference.
  REQUIRE(results.front().spec.method == golden_pipeline::kBestMethod);
  REQUIRE_THAT(results.front().loo_nll, WithinRel(golden_pipeline::kBestLooNll, 1e-9));

  // Full ranking (method order + scores) matches.
  for (size_t i = 0; i < results.size(); ++i) {
    REQUIRE(results[i].spec.method == golden_pipeline::kRankMethods[i]);
    REQUIRE_THAT(results[i].loo_nll, WithinRel(golden_pipeline::kRankLooNll[i], 1e-9));
  }
}
