#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <Eigen/Dense>

#include <stdexcept>

#include "adgencov/projection.hpp"

using adgencov::PairSymmetry;
using adgencov::pair_symmetry_banded;
using adgencov::pair_symmetry_from_generators;
using adgencov::pair_symmetry_from_labels;
using adgencov::reynolds_project;
using Catch::Matchers::WithinAbs;

namespace {
constexpr double kTol = 1e-12;

// True iff P is symmetric to within kTol.
bool is_symmetric(const Eigen::MatrixXd& P) {
  return (P - P.transpose()).cwiseAbs().maxCoeff() <= kTol;
}
}  // namespace

TEST_CASE("projection preserves a 1x1 matrix", "[projection]") {
  Eigen::MatrixXd S(1, 1);
  S << 3.5;
  const Eigen::MatrixXd P = reynolds_project(S, std::vector<int>{0});
  REQUIRE_THAT(P(0, 0), WithinAbs(3.5, kTol));
}

TEST_CASE("singleton blocks leave the matrix unchanged", "[projection]") {
  // Every variable in its own block => projection is the identity map.
  Eigen::MatrixXd S(3, 3);
  S << 2.0, 0.7, -0.3,
       0.7, 1.5,  0.4,
      -0.3, 0.4,  3.0;
  const Eigen::MatrixXd P = reynolds_project(S, {0, 1, 2});
  REQUIRE((P - S).cwiseAbs().maxCoeff() <= kTol);
}

TEST_CASE("single block averages diagonal and off-diagonal separately",
          "[projection]") {
  Eigen::MatrixXd S(3, 3);
  S << 2.0, 0.4, 0.2,
       0.4, 4.0, 0.6,
       0.2, 0.6, 6.0;
  // One block containing all three variables.
  const Eigen::MatrixXd P = reynolds_project(S, {7, 7, 7});

  const double diag_mean = (2.0 + 4.0 + 6.0) / 3.0;          // = 4.0
  const double off_mean = (0.4 + 0.2 + 0.6) / 3.0;           // mean of upper off-diag

  for (int i = 0; i < 3; ++i) {
    REQUIRE_THAT(P(i, i), WithinAbs(diag_mean, kTol));
    for (int j = 0; j < 3; ++j) {
      if (i != j) REQUIRE_THAT(P(i, j), WithinAbs(off_mean, kTol));
    }
  }
  REQUIRE(is_symmetric(P));
}

TEST_CASE("two blocks average within and across correctly", "[projection]") {
  // Variables {0,1} in block A, {2,3} in block B.
  Eigen::MatrixXd S(4, 4);
  S << 1.0, 0.2, 0.5, 0.7,
       0.2, 3.0, 0.9, 0.1,
       0.5, 0.9, 2.0, 0.4,
       0.7, 0.1, 0.4, 4.0;
  const Eigen::MatrixXd P = reynolds_project(S, {0, 0, 1, 1});

  const double diagA = (1.0 + 3.0) / 2.0;   // 2.0
  const double offA = 0.2;                  // only one off-diag pair (symmetric)
  const double diagB = (2.0 + 4.0) / 2.0;   // 3.0
  const double offB = 0.4;
  const double cross = (0.5 + 0.7 + 0.9 + 0.1) / 4.0;  // 0.55

  REQUIRE_THAT(P(0, 0), WithinAbs(diagA, kTol));
  REQUIRE_THAT(P(1, 1), WithinAbs(diagA, kTol));
  REQUIRE_THAT(P(0, 1), WithinAbs(offA, kTol));
  REQUIRE_THAT(P(2, 2), WithinAbs(diagB, kTol));
  REQUIRE_THAT(P(3, 3), WithinAbs(diagB, kTol));
  REQUIRE_THAT(P(2, 3), WithinAbs(offB, kTol));
  for (int i = 0; i < 2; ++i)
    for (int j = 2; j < 4; ++j)
      REQUIRE_THAT(P(i, j), WithinAbs(cross, kTol));
  REQUIRE(is_symmetric(P));
}

TEST_CASE("projection is idempotent (P(P(S)) == P(S))", "[projection]") {
  Eigen::MatrixXd S(4, 4);
  S << 1.0, 0.2, 0.5, 0.7,
       0.2, 3.0, 0.9, 0.1,
       0.5, 0.9, 2.0, 0.4,
       0.7, 0.1, 0.4, 4.0;
  const std::vector<int> labels{0, 0, 1, 1};
  const Eigen::MatrixXd P1 = reynolds_project(S, labels);
  const Eigen::MatrixXd P2 = reynolds_project(P1, labels);
  REQUIRE((P2 - P1).cwiseAbs().maxCoeff() <= kTol);
}

TEST_CASE("projection preserves total trace", "[projection]") {
  Eigen::MatrixXd S(4, 4);
  S << 1.0, 0.2, 0.5, 0.7,
       0.2, 3.0, 0.9, 0.1,
       0.5, 0.9, 2.0, 0.4,
       0.7, 0.1, 0.4, 4.0;
  const Eigen::MatrixXd P = reynolds_project(S, {0, 0, 1, 1});
  REQUIRE_THAT(P.trace(), WithinAbs(S.trace(), kTol));
}

TEST_CASE("mismatched sizes throw", "[projection]") {
  Eigen::MatrixXd S(2, 2);
  S.setIdentity();
  REQUIRE_THROWS_AS(reynolds_project(S, std::vector<int>{0}),
                    std::invalid_argument);
}

// ---------------------------------------------------------------------------
// General orbit-averaging (commutant) projection.
// ---------------------------------------------------------------------------

namespace {
Eigen::MatrixXd sampleS() {
  Eigen::MatrixXd S(4, 4);
  S << 1.0, 0.2, 0.5, 0.7,
       0.2, 3.0, 0.9, 0.1,
       0.5, 0.9, 2.0, 0.4,
       0.7, 0.1, 0.4, 4.0;
  return S;
}
}  // namespace

TEST_CASE("pair_symmetry_from_labels reproduces the block projection",
          "[projection][symmetry]") {
  const Eigen::MatrixXd S = sampleS();
  const std::vector<int> labels{0, 0, 1, 1};
  const PairSymmetry sym = pair_symmetry_from_labels(labels);
  // The generic orbit average must match the hand-written block projection to
  // round-off — same orbits, only the summation order differs.
  const Eigen::MatrixXd a = reynolds_project(S, labels);
  const Eigen::MatrixXd b = reynolds_project(S, sym);
  REQUIRE((a - b).cwiseAbs().maxCoeff() <= kTol);
}

TEST_CASE("cyclic group yields a circulant projection", "[projection][symmetry]") {
  const Eigen::MatrixXd S = sampleS();
  // C_4 generated by the shift i -> (i+1) mod 4.
  const PairSymmetry sym = pair_symmetry_from_generators(4, {{1, 2, 3, 0}});
  const Eigen::MatrixXd P = reynolds_project(S, sym);
  REQUIRE(is_symmetric(P));
  // Circulant: entry depends only on (j - i) mod p, i.e. P is shift-invariant.
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      REQUIRE_THAT(P(i, j), WithinAbs(P((i + 1) % 4, (j + 1) % 4), kTol));
  // Trace is preserved (projection is orthogonal and identity-preserving).
  REQUIRE_THAT(P.trace(), WithinAbs(S.trace(), kTol));
}

TEST_CASE("banded symmetry averages along positional distance",
          "[projection][symmetry]") {
  const Eigen::MatrixXd S = sampleS();
  const PairSymmetry sym = pair_symmetry_banded(4);
  REQUIRE(sym.n_orbits == 4);
  const Eigen::MatrixXd P = reynolds_project(S, sym);
  REQUIRE(is_symmetric(P));
  // P(i,j) depends only on |i-j|; check band 1 is the mean of S's band 1.
  const double band1 = (S(0, 1) + S(1, 2) + S(2, 3)) / 3.0;
  REQUIRE_THAT(P(0, 1), WithinAbs(band1, kTol));
  REQUIRE_THAT(P(1, 2), WithinAbs(band1, kTol));
  REQUIRE_THAT(P(2, 3), WithinAbs(band1, kTol));
  // Diagonal (band 0) is the mean of the diagonal.
  const double band0 = (S(0, 0) + S(1, 1) + S(2, 2) + S(3, 3)) / 4.0;
  for (int i = 0; i < 4; ++i) REQUIRE_THAT(P(i, i), WithinAbs(band0, kTol));
}

TEST_CASE("orbit projection is symmetric and idempotent",
          "[projection][symmetry]") {
  const Eigen::MatrixXd S = sampleS();
  const PairSymmetry sym = pair_symmetry_from_generators(4, {{1, 2, 3, 0}});
  const Eigen::MatrixXd P1 = reynolds_project(S, sym);
  const Eigen::MatrixXd P2 = reynolds_project(P1, sym);
  REQUIRE((P2 - P1).cwiseAbs().maxCoeff() <= kTol);
}

TEST_CASE("from_generators rejects a non-permutation", "[projection][symmetry]") {
  REQUIRE_THROWS_AS(pair_symmetry_from_generators(4, {{0, 1, 1, 3}}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pair_symmetry_from_generators(3, {{0, 1}}),
                    std::invalid_argument);
}
