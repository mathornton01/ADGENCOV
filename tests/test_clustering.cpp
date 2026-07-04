// Tests for average-linkage agglomerative clustering and the data-driven
// symmetry groups (correlation_blocks, hierarchical_wreath).  Partitions are
// checked against sklearn/prototype goldens from gen_golden_clustering.py.
//
// Cluster ids are arbitrary, so equality is tested at the PARTITION level:
// two labelings agree iff factorize() (dense codes in first-appearance order)
// maps them to the identical sequence.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "adgencov/clustering.hpp"
#include "adgencov/groups.hpp"
#include "adgencov/io.hpp"
#include "adgencov/preprocess.hpp"

#include "golden_clustering.hpp"

#ifndef ADGENCOV_TEST_DIR
#define ADGENCOV_TEST_DIR "."
#endif

using namespace adgencov;

namespace {

Dataset load_fixture() {
  const std::string fixture = std::string(ADGENCOV_TEST_DIR) + "/fixtures/expr_fixture.tsv";
  ExpressionData raw = load_expression_matrix(fixture, "_LL[0-9]+", "gene_short_name");
  return preprocess(raw, /*n_genes=*/6, /*min_mean=*/0.1, /*log_transform=*/true);
}

// Canonical partition signature: dense codes in first-appearance order.
std::vector<int> canon(const std::vector<std::string>& labels) { return factorize(labels); }
std::vector<int> canon(const std::vector<int>& labels) {
  std::vector<std::string> s;
  s.reserve(labels.size());
  for (int v : labels) s.push_back(std::to_string(v));
  return factorize(s);
}

Eigen::MatrixXd golden_D() {
  const int n = golden_clustering::kDdim;
  Eigen::MatrixXd D(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) D(i, j) = golden_clustering::kD[i * n + j];
  return D;
}

int n_unique(const std::vector<int>& v) {
  const std::vector<int> c = canon(v);
  if (c.empty()) return 0;
  return *std::max_element(c.begin(), c.end()) + 1;
}

int n_unique(const std::vector<std::string>& v) {
  const std::vector<int> c = canon(v);
  if (c.empty()) return 0;
  return *std::max_element(c.begin(), c.end()) + 1;
}

}  // namespace

TEST_CASE("agglomerative_average matches sklearn on precomputed distances", "[clustering]") {
  const Eigen::MatrixXd D = golden_D();
  const std::vector<int> got = agglomerative_average(D, golden_clustering::kDnClusters);

  // Same partition as sklearn (up to id relabeling).
  REQUIRE(canon(got) == canon(golden_clustering::kDLabels));

  // Structural check independent of the golden: {0,1}, {2,3}, {4}.
  REQUIRE(got.size() == 5u);
  REQUIRE(got[0] == got[1]);
  REQUIRE(got[2] == got[3]);
  REQUIRE(got[0] != got[2]);
  REQUIRE(got[4] != got[0]);
  REQUIRE(got[4] != got[2]);
  REQUIRE(n_unique(got) == 3);
}

TEST_CASE("agglomerative_average id labeling is canonical by smallest member", "[clustering]") {
  const Eigen::MatrixXd D = golden_D();
  const std::vector<int> got = agglomerative_average(D, 3);
  // Cluster containing row 0 gets id 0; the {2,3} cluster gets id 1; loner id 2.
  REQUIRE(got == std::vector<int>({0, 0, 1, 1, 2}));
}

TEST_CASE("agglomerative_average handles the degenerate cluster counts", "[clustering]") {
  const Eigen::MatrixXd D = golden_D();
  const int n = golden_clustering::kDdim;

  // One cluster per point.
  const std::vector<int> singletons = agglomerative_average(D, n);
  REQUIRE(n_unique(singletons) == n);

  // A single cluster swallows everything.
  const std::vector<int> one = agglomerative_average(D, 1);
  REQUIRE(n_unique(one) == 1);
}

TEST_CASE("agglomerative_average validates its arguments", "[clustering]") {
  const Eigen::MatrixXd D = golden_D();
  REQUIRE_THROWS_AS(agglomerative_average(D, 0), std::invalid_argument);
  REQUIRE_THROWS_AS(agglomerative_average(D, golden_clustering::kDdim + 1), std::invalid_argument);
  Eigen::MatrixXd nonsquare(3, 4);
  nonsquare.setZero();
  REQUIRE_THROWS_AS(agglomerative_average(nonsquare, 2), std::invalid_argument);
}

TEST_CASE("correlation_blocks reproduces the prototype partition", "[groups][clustering]") {
  Dataset ds = load_fixture();
  REQUIRE(ds.genes == golden_clustering::kGenes);  // same selection/order as golden

  std::vector<std::string> labels =
      build_group_labels(ds, "correlation_blocks", nullptr, nullptr, golden_clustering::kNBlocks);

  REQUIRE(labels.size() == ds.genes.size());
  REQUIRE(canon(labels) == canon(golden_clustering::kCorrBlocks));
  // The prototype resolves the fixture into 4 blocks.
  REQUIRE(n_unique(labels) == golden_clustering::kNBlocks);
}

TEST_CASE("correlation_blocks rejects an out-of-range block count", "[groups][clustering]") {
  Dataset ds = load_fixture();
  REQUIRE_THROWS_AS(build_group_labels(ds, "correlation_blocks", nullptr, nullptr, 0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      build_group_labels(ds, "correlation_blocks", nullptr, nullptr,
                         static_cast<int>(ds.genes.size()) + 1),
      std::invalid_argument);
}

TEST_CASE("hierarchical_wreath nests correlation blocks within mapped groups",
          "[groups][clustering]") {
  Dataset ds = load_fixture();
  const std::string gmpath = std::string(ADGENCOV_TEST_DIR) + "/fixtures/group_map_fixture.tsv";
  Table gmap = read_table(gmpath);

  std::vector<std::string> labels =
      build_group_labels(ds, "hierarchical_wreath", nullptr, &gmap, golden_clustering::kNBlocks);

  REQUIRE(labels.size() == ds.genes.size());
  REQUIRE(canon(labels) == canon(golden_clustering::kWreath));
  for (const auto& l : labels) REQUIRE(l.find("::") != std::string::npos);  // coarse::fine
}

TEST_CASE("hierarchical_wreath requires a group map", "[groups][clustering]") {
  Dataset ds = load_fixture();
  REQUIRE_THROWS_AS(
      build_group_labels(ds, "hierarchical_wreath", nullptr, nullptr, golden_clustering::kNBlocks),
      std::invalid_argument);
}
