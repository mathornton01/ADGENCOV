// Recommender / likelihood unit tests for ADGENCOV.
//
// Validates the selection layer — estimate_covariance dispatcher,
// gaussian_nll_one, the rank-1-downdate leave-one-out NLL, candidate_grid and
// recommend_estimator — against golden values produced by
// tests/gen_golden_select.py from the ORIGINAL prototype (ad_covariance_app.py).
//
// Beyond the Python goldens, two internal-consistency checks stand on their own:
//   * the rank-1 downdate reproduces np.cov(train) exactly, and
//   * the downdate-based loo_nll equals a brute-force leave-one-out that rebuilds
//     each training submatrix from scratch.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "adgencov/projection.hpp"
#include "adgencov/select.hpp"
#include "adgencov/shrink.hpp"
#include "golden_select.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace adgencov;

namespace {

constexpr double kTol = 1e-9;

Eigen::MatrixXd fixtureX() {
  Eigen::MatrixXd X(golden_select::kN, golden_select::kP);
  for (int i = 0; i < golden_select::kN; ++i)
    for (int j = 0; j < golden_select::kP; ++j)
      X(i, j) = golden_select::kX[static_cast<std::size_t>(i) * golden_select::kP + j];
  return X;
}

std::vector<int> fixtureLabels() { return golden_select::kLabels; }

Eigen::MatrixXd toEigen(const golden_select::GoldenMat& g) {
  Eigen::MatrixXd M(g.rows, g.cols);
  for (int i = 0; i < g.rows; ++i)
    for (int j = 0; j < g.cols; ++j)
      M(i, j) = g.data[static_cast<std::size_t>(i) * g.cols + j];
  return M;
}

Eigen::VectorXd toVec(const std::vector<double>& v) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(v.size()));
  for (std::size_t i = 0; i < v.size(); ++i) out(static_cast<Eigen::Index>(i)) = v[i];
  return out;
}

Eigen::MatrixXd toSquare(const std::vector<double>& v, int p) {
  Eigen::MatrixXd M(p, p);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j < p; ++j)
      M(i, j) = v[static_cast<std::size_t>(i) * p + j];
  return M;
}

void expectClose(const Eigen::MatrixXd& got, const golden_select::GoldenMat& want,
                 double tol = kTol) {
  const Eigen::MatrixXd expected = toEigen(want);
  REQUIRE(got.rows() == expected.rows());
  REQUIRE(got.cols() == expected.cols());
  const double err = (got - expected).cwiseAbs().maxCoeff();
  INFO("max abs entrywise error = " << err);
  REQUIRE(err <= tol);
}

// String signature of a spec, matching gen_golden_select.py's sig().
std::string sig(const EstimatorSpec& s) {
  if (s.params.empty()) return s.method;
  std::string out = s.method + "[";
  bool first = true;
  for (const auto& kv : s.params) {  // std::map iterates in sorted key order
    if (!first) out += ",";
    first = false;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%g", kv.second);
    out += kv.first + "=" + buf;
  }
  out += "]";
  return out;
}

// Rank-1 downdate of the sample covariance with row i removed (== np.cov(train)).
Eigen::MatrixXd downdateCov(const Eigen::MatrixXd& X, int i) {
  const double n = static_cast<double>(X.rows());
  const Eigen::RowVectorXd xbar = X.colwise().mean();
  const Eigen::MatrixXd Xc = X.rowwise() - xbar;
  const Eigen::MatrixXd C = Xc.transpose() * Xc;
  const Eigen::RowVectorXd di = X.row(i) - xbar;
  const Eigen::MatrixXd Ci = C - (n / (n - 1.0)) * (di.transpose() * di);
  return Ci / (n - 2.0);
}

// Brute-force leave-one-out NLL that rebuilds each training submatrix.
double bruteLoo(const Eigen::MatrixXd& X, const std::vector<int>& labels,
                const EstimatorSpec& spec) {
  const Eigen::Index n = X.rows(), p = X.cols();
  double total = 0.0;
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::MatrixXd Xtr(n - 1, p);
    Xtr.topRows(i) = X.topRows(i);
    Xtr.bottomRows(n - 1 - i) = X.bottomRows(n - 1 - i);
    const Eigen::VectorXd mu = Xtr.colwise().mean().transpose();
    const Eigen::MatrixXd Sigma = estimate_covariance(Xtr, labels, spec);
    total += gaussian_nll_one(X.row(i).transpose(), mu, Sigma);
  }
  return total / static_cast<double>(n);
}

}  // namespace

// --- gaussian_nll_one --------------------------------------------------------

TEST_CASE("gaussian_nll_one matches the prototype", "[select][nll]") {
  const Eigen::MatrixXd Sigma = toSquare(golden_select::kNllSigma, golden_select::kP);
  const Eigen::VectorXd mu = toVec(golden_select::kNllMu);
  const Eigen::VectorXd x = toVec(golden_select::kNllX);
  const double got = gaussian_nll_one(x, mu, Sigma);
  REQUIRE_THAT(got, WithinAbs(golden_select::kNllValue, kTol));
}

TEST_CASE("gaussian_nll_one on identity covariance is analytic", "[select][nll]") {
  // Sigma = I_3, mu = 0, x = (1,0,0)  =>  0.5*(3*log(2*pi) + 0 + 1).
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd x(3);
  x << 1.0, 0.0, 0.0;
  const double expected = 0.5 * (3.0 * std::log(2.0 * M_PI) + 0.0 + 1.0);
  REQUIRE_THAT(gaussian_nll_one(x, mu, I), WithinAbs(expected, 1e-12));
}

// --- Rank-1 downdate ---------------------------------------------------------

TEST_CASE("rank-1 downdate reproduces np.cov(train) exactly", "[select][downdate]") {
  const Eigen::MatrixXd X = fixtureX();
  expectClose(downdateCov(X, 0), golden_select::loo_cov_drop0);
  expectClose(downdateCov(X, 3), golden_select::loo_cov_drop3);
}

TEST_CASE("downdate matches an explicit submatrix sample_covariance",
          "[select][downdate]") {
  const Eigen::MatrixXd X = fixtureX();
  const int n = static_cast<int>(X.rows());
  for (int i : {0, 2, 5}) {
    Eigen::MatrixXd Xtr(n - 1, X.cols());
    Xtr.topRows(i) = X.topRows(i);
    Xtr.bottomRows(n - 1 - i) = X.bottomRows(n - 1 - i);
    const Eigen::MatrixXd direct = sample_covariance(Xtr, /*unbiased=*/true);
    REQUIRE((downdateCov(X, i) - direct).cwiseAbs().maxCoeff() <= kTol);
  }
}

// --- estimate_covariance dispatcher ------------------------------------------

TEST_CASE("estimate_covariance dispatches every method to the reference",
          "[select][dispatch]") {
  const Eigen::MatrixXd X = fixtureX();
  const auto labels = fixtureLabels();
  expectClose(estimate_covariance(X, labels, {"ad_ridge", {{"alpha", 0.2}}}),
              golden_select::est_ad_ridge_0p2);
  expectClose(estimate_covariance(X, labels, {"ad_lasso", {{"lam", 0.1}}}),
              golden_select::est_ad_lasso_0p1);
  expectClose(estimate_covariance(X, labels, {"lw", {}}), golden_select::est_lw);
  expectClose(estimate_covariance(X, labels, {"oas", {}}), golden_select::est_oas);
  expectClose(estimate_covariance(X, labels, {"ad_linear_lw", {}}),
              golden_select::est_ad_linear_lw);
  expectClose(estimate_covariance(X, labels, {"ad_oas", {}}),
              golden_select::est_ad_oas);
}

TEST_CASE("estimate_covariance always returns an SPD matrix", "[select][dispatch]") {
  const Eigen::MatrixXd X = fixtureX();
  const auto labels = fixtureLabels();
  for (const auto& spec : candidate_grid(golden_select::kP, golden_select::kN)) {
    const Eigen::MatrixXd Sigma = estimate_covariance(X, labels, spec);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
    INFO("method = " << spec.method);
    REQUIRE(es.info() == Eigen::Success);
    REQUIRE(es.eigenvalues().minCoeff() > 0.0);
  }
}

TEST_CASE("estimate_covariance rejects an unknown method", "[select][dispatch]") {
  const Eigen::MatrixXd X = fixtureX();
  REQUIRE_THROWS(estimate_covariance(X, fixtureLabels(), {"nonsense", {}}));
}

// --- loo_nll -----------------------------------------------------------------

TEST_CASE("loo_nll matches the prototype for each method", "[select][loo]") {
  const Eigen::MatrixXd X = fixtureX();
  const auto labels = fixtureLabels();
  const std::vector<EstimatorSpec> specs = {
      {"ad_ridge", {{"alpha", 0.2}}},
      {"ad_lasso", {{"lam", 0.1}}},
      {"ad_elastic_net", {{"lam", 0.1}, {"l1_ratio", 0.25}}},
      {"lw", {}},
      {"oas", {}},
      {"ad_linear_lw", {}},
      {"ad_oas", {}},
  };
  REQUIRE(specs.size() == golden_select::kLooSigs.size());
  for (std::size_t k = 0; k < specs.size(); ++k) {
    INFO("spec = " << sig(specs[k]));
    REQUIRE(sig(specs[k]) == golden_select::kLooSigs[k]);  // guard ordering
    REQUIRE_THAT(loo_nll(X, labels, specs[k]),
                 WithinAbs(golden_select::kLooVals[k], kTol));
  }
}

TEST_CASE("downdate loo_nll equals brute-force leave-one-out", "[select][loo]") {
  const Eigen::MatrixXd X = fixtureX();
  const auto labels = fixtureLabels();
  // Covariance-only methods take the fast downdate path; it must equal the
  // naive path that rebuilds each training submatrix.
  for (const EstimatorSpec spec : {
           EstimatorSpec{"ad_ridge", {{"alpha", 0.2}}},
           EstimatorSpec{"ad_lasso", {{"lam", 0.1}}},
           EstimatorSpec{"ad_elastic_net", {{"lam", 0.1}, {"l1_ratio", 0.25}}},
           EstimatorSpec{"ridge", {{"alpha", 0.3}}},
       }) {
    INFO("spec = " << sig(spec));
    REQUIRE_THAT(loo_nll(X, labels, spec),
                 WithinAbs(bruteLoo(X, labels, spec), 1e-11));
  }
}

TEST_CASE("loo_nll rejects too-few samples", "[select][loo]") {
  Eigen::MatrixXd X(2, 3);
  X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  REQUIRE_THROWS(loo_nll(X, {0, 0, 0}, {"ridge", {{"alpha", 0.2}}}));
}

// --- candidate_grid & recommend_estimator ------------------------------------

TEST_CASE("candidate_grid has the expected shape", "[select][grid]") {
  const auto grid = candidate_grid(5, 6);
  // 5 ridge + 4 data-driven + 4*(lasso+elastic_net) = 17 candidates.
  REQUIRE(grid.size() == 17);
}

TEST_CASE("recommend_estimator ranking matches the prototype", "[select][recommend]") {
  const Eigen::MatrixXd X = fixtureX();
  const auto results = recommend_estimator(X, fixtureLabels());

  REQUIRE(results.size() == golden_select::kRecSigs.size());
  for (std::size_t k = 0; k < results.size(); ++k) {
    INFO("rank " << k << " got " << sig(results[k].spec)
                 << " want " << golden_select::kRecSigs[k]);
    REQUIRE(sig(results[k].spec) == golden_select::kRecSigs[k]);
    REQUIRE_THAT(results[k].loo_nll, WithinAbs(golden_select::kRecLoo[k], kTol));
    REQUIRE_THAT(results[k].condition_number,
                 WithinRel(golden_select::kRecCond[k], 1e-7));
  }
}

TEST_CASE("recommend_estimator returns a strictly sorted, finite ranking",
          "[select][recommend]") {
  const Eigen::MatrixXd X = fixtureX();
  const auto results = recommend_estimator(X, fixtureLabels());
  REQUIRE(!results.empty());
  for (std::size_t k = 1; k < results.size(); ++k) {
    REQUIRE(results[k - 1].loo_nll <= results[k].loo_nll);
  }
  for (const auto& r : results) {
    REQUIRE(std::isfinite(r.loo_nll));
    REQUIRE(std::isfinite(r.condition_number));
  }
}
