#include "adgencov/select.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "adgencov/projection.hpp"
#include "adgencov/shrink.hpp"

namespace adgencov {

namespace {

constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

// Fetch a scalar hyper-parameter with a default, matching params.get(key, def).
double param(const EstimatorSpec& spec, const std::string& key, double def) {
  const auto it = spec.params.find(key);
  return it == spec.params.end() ? def : it->second;
}

bool is_ad(const std::string& method) {
  return method.rfind("ad_", 0) == 0;  // starts with "ad_"
}

// Whether the method's estimate depends on the raw data matrix X (Ledoit-Wolf /
// OAS shrinkage intensities), as opposed to only the sample covariance.
bool needs_data_matrix(const std::string& m) {
  return m == "lw" || m == "ledoit_wolf" || m == "oas" ||
         m == "ad_linear_lw" || m == "ad_oas";
}

// Dispatch the covariance-only estimators given an already-formed (and, for AD
// variants, already Reynolds-projected) covariance S0.  Returns the make_pd'd
// estimate, exactly as the prototype's estimate_covariance does.
Eigen::MatrixXd dispatch_linear(const Eigen::MatrixXd& S0,
                                const EstimatorSpec& spec) {
  const std::string& m = spec.method;
  if (m == "sample" || m == "ad_sample") {
    return make_pd(S0);
  }
  if (m == "ridge" || m == "ad_ridge") {
    return make_pd(ridge(S0, param(spec, "alpha", 0.2)));
  }
  if (m == "lasso" || m == "ad_lasso") {
    return make_pd(soft_threshold_offdiag(S0, param(spec, "lam", 0.05), 1.0));
  }
  if (m == "elastic_net" || m == "ad_elastic_net") {
    return make_pd(soft_threshold_offdiag(S0, param(spec, "lam", 0.05),
                                          param(spec, "l1_ratio", 0.25)));
  }
  throw std::invalid_argument("estimate_covariance: unknown method '" + m + "'");
}

}  // namespace

Eigen::MatrixXd estimate_covariance(const Eigen::MatrixXd& X,
                                    const std::vector<int>& labels,
                                    const EstimatorSpec& spec) {
  if (static_cast<Eigen::Index>(labels.size()) != X.cols()) {
    throw std::invalid_argument(
        "estimate_covariance: labels length must equal number of genes (cols)");
  }
  const std::string& m = spec.method;

  // Data-driven shrinkers computed directly from X.
  if (m == "lw" || m == "ledoit_wolf") {
    return make_pd(ledoit_wolf(X));
  }
  if (m == "oas") {
    return make_pd(oas(X));
  }

  const Eigen::MatrixXd S = sample_covariance(X, /*unbiased=*/true);
  const Eigen::MatrixXd S0 = is_ad(m) ? reynolds_project(S, labels) : S;

  if (m == "ad_linear_lw") {
    return make_pd(ridge(S0, ledoit_wolf_shrinkage(X)));
  }
  if (m == "ad_oas") {
    return make_pd(ridge(S0, oas_shrinkage(X)));
  }
  return dispatch_linear(S0, spec);
}

double gaussian_nll_one(const Eigen::VectorXd& x, const Eigen::VectorXd& mu,
                        const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = x.size();
  // Match the prototype: project onto SPD before evaluating the likelihood.
  const Eigen::MatrixXd Spd = make_pd(Sigma);
  const Eigen::LLT<Eigen::MatrixXd> llt(Spd);
  if (llt.info() != Eigen::Success) {
    return std::numeric_limits<double>::infinity();
  }
  // logdet(Sigma) = 2 * sum(log(diag(L))) with Sigma = L L^T.
  const Eigen::MatrixXd& L = llt.matrixL();
  double logdet = 0.0;
  for (Eigen::Index i = 0; i < p; ++i) {
    logdet += std::log(L(i, i));
  }
  logdet *= 2.0;
  // q = (x-mu)^T Sigma^{-1} (x-mu) = || L^{-1} (x-mu) ||^2.
  const Eigen::VectorXd d = x - mu;
  const Eigen::VectorXd y = L.triangularView<Eigen::Lower>().solve(d);
  const double q = y.squaredNorm();
  return 0.5 * (static_cast<double>(p) * std::log(kTwoPi) + logdet + q);
}

double loo_nll(const Eigen::MatrixXd& X, const std::vector<int>& labels,
               const EstimatorSpec& spec) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (n < 3) {
    throw std::invalid_argument(
        "loo_nll: need >= 3 samples for a leave-one-out unbiased covariance");
  }
  const double dn = static_cast<double>(n);

  const Eigen::RowVectorXd xbar = X.colwise().mean();
  const Eigen::MatrixXd Xc = X.rowwise() - xbar;
  const Eigen::MatrixXd C = Xc.transpose() * Xc;  // full scatter about full mean

  const std::string& m = spec.method;
  const bool needs_X = needs_data_matrix(m);
  const bool ad = is_ad(m);

  double total = 0.0;
  for (Eigen::Index i = 0; i < n; ++i) {
    // Leave-one-out training mean: (n*xbar - x_i) / (n-1).
    const Eigen::RowVectorXd xi = X.row(i);
    const Eigen::VectorXd mu_i =
        ((dn * xbar - xi) / (dn - 1.0)).transpose();

    Eigen::MatrixXd Sigma;
    if (needs_X) {
      // Build the (n-1) x p training submatrix; these estimators depend on X.
      Eigen::MatrixXd Xtr(n - 1, p);
      Xtr.topRows(i) = X.topRows(i);
      Xtr.bottomRows(n - 1 - i) = X.bottomRows(n - 1 - i);
      Sigma = estimate_covariance(Xtr, labels, spec);
    } else {
      // Exact rank-1 downdate of the scatter about the leave-one-out mean:
      //   C_i = C - (n/(n-1)) (x_i - xbar)(x_i - xbar)^T,
      //   S_i = C_i / (n-2)  == np.cov(train, ddof=1).
      const Eigen::RowVectorXd di = xi - xbar;
      const Eigen::MatrixXd Ci =
          C - (dn / (dn - 1.0)) * (di.transpose() * di);
      const Eigen::MatrixXd S_i = Ci / (dn - 2.0);
      const Eigen::MatrixXd S0 = ad ? reynolds_project(S_i, labels) : S_i;
      Sigma = dispatch_linear(S0, spec);
    }
    total += gaussian_nll_one(xi.transpose(), mu_i, Sigma);
  }
  return total / dn;
}

std::vector<EstimatorSpec> candidate_grid(int /*p*/, int /*n*/) {
  std::vector<EstimatorSpec> grid;
  for (double a : {0.05, 0.1, 0.2, 0.4, 0.7}) {
    grid.push_back({"ad_ridge", {{"alpha", a}}});
  }
  grid.push_back({"ad_linear_lw", {}});
  grid.push_back({"ad_oas", {}});
  grid.push_back({"lw", {}});
  grid.push_back({"oas", {}});
  for (double lam : {0.01, 0.03, 0.1, 0.3}) {
    grid.push_back({"ad_lasso", {{"lam", lam}}});
    grid.push_back({"ad_elastic_net", {{"lam", lam}, {"l1_ratio", 0.25}}});
  }
  return grid;
}

std::vector<EstimatorResult> recommend_estimator(
    const Eigen::MatrixXd& X, const std::vector<int>& labels) {
  const auto grid = candidate_grid(static_cast<int>(X.cols()),
                                   static_cast<int>(X.rows()));
  std::vector<EstimatorResult> results;
  results.reserve(grid.size());

  for (const auto& spec : grid) {
    try {
      const double score = loo_nll(X, labels, spec);
      const Eigen::MatrixXd Sigma = estimate_covariance(X, labels, spec);
      // 2-norm condition number of an SPD matrix = lambda_max / lambda_min.
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
      double cond = std::numeric_limits<double>::infinity();
      if (es.info() == Eigen::Success) {
        const Eigen::VectorXd ev = es.eigenvalues().cwiseAbs();
        const double lo = ev.minCoeff();
        cond = lo > 0.0 ? ev.maxCoeff() / lo
                        : std::numeric_limits<double>::infinity();
      }
      results.push_back({spec, Sigma, score, cond});
    } catch (const std::exception&) {
      // Match the prototype: skip candidates that fail to estimate.
    }
  }

  std::stable_sort(results.begin(), results.end(),
                   [](const EstimatorResult& a, const EstimatorResult& b) {
                     return a.loo_nll < b.loo_nll;
                   });
  return results;
}

}  // namespace adgencov
