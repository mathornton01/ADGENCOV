#include "adgencov/select.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "adgencov/projection.hpp"
#include "adgencov/shrink.hpp"

namespace adgencov {

namespace {

constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

// A projection operator applied to a p-by-p sample covariance.  The block
// (labels) path and the general-symmetry (PairSymmetry) path differ only in
// which projector they bind here; all estimator logic below is projector-
// agnostic, so both paths run identical numerics.
using ProjectFn = std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;

// Fetch a scalar hyper-parameter with a default, matching params.get(key, def).
double param(const EstimatorSpec& spec, const std::string& key, double def) {
  const auto it = spec.params.find(key);
  return it == spec.params.end() ? def : it->second;
}

bool is_ad(const std::string& method) {
  return method.rfind("ad_", 0) == 0;  // starts with "ad_"
}

// Whether the method's estimate depends on the raw data matrix X (Ledoit-Wolf /
// OAS shrinkage intensities), as opposed to only the sample covariance.  These
// take the submatrix path in loo_nll; everything else uses the rank-1 downdate.
bool needs_data_matrix(const std::string& m) {
  return m == "lw" || m == "ledoit_wolf" || m == "oas" ||
         m == "ad_linear_lw" || m == "ad_oas" ||
         m == "ad_target_lw" || m == "ad_target_oas" ||
         m == "ad_target_optimal";
}

// Dispatch the covariance-only estimators from a raw sample covariance S and a
// projector.  Projection-first AD variants project S first; the symmetry-target
// ad_target_ridge shrinks S toward the projected target P_G(S).  Returns the
// make_pd'd estimate, exactly as the prototype's estimate_covariance does.
Eigen::MatrixXd dispatch_cov(const Eigen::MatrixXd& S, const ProjectFn& project,
                             const EstimatorSpec& spec) {
  const std::string& m = spec.method;

  // Symmetry-target: (1-lam) S + lam P_G(S), optional small identity ridge.
  if (m == "ad_target_ridge") {
    const Eigen::MatrixXd PG = project(S);
    Eigen::MatrixXd C = shrink_to_target(S, PG, param(spec, "lam", 0.5));
    const double diag_alpha = param(spec, "diag_alpha", 1.0e-3);
    if (diag_alpha > 0.0) C = ridge(C, diag_alpha);
    return make_pd(C);
  }

  // Projection-first AD variants share the projected covariance S0; the plain
  // linear/sparse estimators use S0 = S.
  const Eigen::MatrixXd S0 = is_ad(m) ? project(S) : S;
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

// Full estimator dispatch from data X and a projector: the X-dependent
// shrinkers (Ledoit-Wolf / OAS and their AD / AD-target variants) plus the
// covariance-only family via dispatch_cov.
Eigen::MatrixXd estimate_with_projector(const Eigen::MatrixXd& X,
                                        const ProjectFn& project,
                                        const EstimatorSpec& spec) {
  const std::string& m = spec.method;

  // Data-driven shrinkers computed directly from X.
  if (m == "lw" || m == "ledoit_wolf") {
    return make_pd(ledoit_wolf(X));
  }
  if (m == "oas") {
    return make_pd(oas(X));
  }

  const Eigen::MatrixXd S = sample_covariance(X, /*unbiased=*/true);

  // Projection-first AD shrinkers: project, then ridge by the LW/OAS intensity.
  if (m == "ad_linear_lw") {
    return make_pd(ridge(project(S), ledoit_wolf_shrinkage(X)));
  }
  if (m == "ad_oas") {
    return make_pd(ridge(project(S), oas_shrinkage(X)));
  }

  // Symmetry-target shrinkers driven by the LW/OAS intensity: shrink S toward
  // the projected target P_G(S) with strength alpha = LW/OAS(X).
  if (m == "ad_target_lw" || m == "ad_target_oas") {
    const double lambda =
        (m == "ad_target_lw") ? ledoit_wolf_shrinkage(X) : oas_shrinkage(X);
    Eigen::MatrixXd C = shrink_to_target(S, project(S), lambda);
    const double diag_alpha = param(spec, "diag_alpha", 0.0);
    if (diag_alpha > 0.0) C = ridge(C, diag_alpha);
    return make_pd(C);
  }

  // Symmetry-aware OPTIMAL convex shrinkage (Thornton, "Symmetry-Aware Convex
  // Shrinkage", arXiv:2605.17111, Prop. 3.2 / Eqs. 18-20).  Unlike ad_target_lw
  // /oas (which reuse the identity-target LW/OAS intensity), this derives the
  // population-optimal weight alpha* = V_perp / (V_perp + D) for shrinking the
  // sample covariance toward its group projection, from a data-driven plug-in.
  // All quantities use the BIASED sample covariance R_hat (paper Eq. 8):
  //   R_G     = P_G(R_hat)
  //   V_perp  = (1/n^2) sum_k || P_G^perp(x_k x_k^T) - P_G^perp(R_hat) ||_F^2   (Eq. 18)
  //   V_perp+D = || R_hat - R_G ||_F^2                                          (Eq. 19)
  //   alpha*  = clip( V_perp / (V_perp+D), 0, 1 )                               (Eq. 20)
  //   Sigma   = (1 - alpha*) R_hat + alpha* R_G                                 (Eq. 11)
  if (m == "ad_target_optimal") {
    const Eigen::Index n = X.rows();
    if (n < 1) throw std::invalid_argument("ad_target_optimal: empty X");
    const Eigen::RowVectorXd xbar = X.colwise().mean();
    const Eigen::MatrixXd Xc = X.rowwise() - xbar;
    const double dn = static_cast<double>(n);
    const Eigen::MatrixXd Rhat = (Xc.transpose() * Xc) / dn;   // biased, Eq. 8
    const Eigen::MatrixXd Rg = project(Rhat);
    const Eigen::MatrixXd perpR = Rhat - Rg;                   // P_G^perp(R_hat)
    const double denom = perpR.squaredNorm();                 // Eq. 19 (||.||_F^2)

    double vperp = 0.0;                                        // Eq. 18
    for (Eigen::Index k = 0; k < n; ++k) {
      const Eigen::VectorXd xk = Xc.row(k).transpose();
      const Eigen::MatrixXd dk = xk * xk.transpose();         // x_k x_k^T (centered)
      const Eigen::MatrixXd perp_dk = dk - project(dk);       // P_G^perp(x_k x_k^T)
      vperp += (perp_dk - perpR).squaredNorm();
    }
    vperp /= dn * dn;

    double alpha = (denom > 0.0) ? (vperp / denom) : 0.0;     // Eq. 20
    alpha = std::min(1.0, std::max(0.0, alpha));
    Eigen::MatrixXd C = (1.0 - alpha) * Rhat + alpha * Rg;    // Eq. 11
    const double diag_alpha = param(spec, "diag_alpha", 0.0);
    if (diag_alpha > 0.0) C = ridge(C, diag_alpha);
    return make_pd(C);
  }

  return dispatch_cov(S, project, spec);
}

// Leave-one-out CV mean NLL given data X and a projector (see the header for the
// downdate / submatrix split).  Both public overloads bind `project` and call
// this, so labels and general-symmetry paths run identical code.
double loo_with_projector(const Eigen::MatrixXd& X, const ProjectFn& project,
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

  const bool needs_X = needs_data_matrix(spec.method);

  double total = 0.0;
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::RowVectorXd xi = X.row(i);
    const Eigen::VectorXd mu_i = ((dn * xbar - xi) / (dn - 1.0)).transpose();

    Eigen::MatrixXd Sigma;
    if (needs_X) {
      // Build the (n-1) x p training submatrix; these estimators depend on X.
      Eigen::MatrixXd Xtr(n - 1, p);
      Xtr.topRows(i) = X.topRows(i);
      Xtr.bottomRows(n - 1 - i) = X.bottomRows(n - 1 - i);
      Sigma = estimate_with_projector(Xtr, project, spec);
    } else {
      // Exact rank-1 downdate of the scatter about the leave-one-out mean:
      //   C_i = C - (n/(n-1)) (x_i - xbar)(x_i - xbar)^T,  S_i = C_i / (n-2).
      const Eigen::RowVectorXd di = xi - xbar;
      const Eigen::MatrixXd Ci = C - (dn / (dn - 1.0)) * (di.transpose() * di);
      const Eigen::MatrixXd S_i = Ci / (dn - 2.0);
      Sigma = dispatch_cov(S_i, project, spec);
    }
    total += gaussian_nll_one(xi.transpose(), mu_i, Sigma);
  }
  return total / dn;
}

// True for the projection-first AD estimators whose covariance is *constant on
// the symmetry orbits* (ad_ridge, ad_sample, ad_lasso, ad_elastic_net,
// ad_linear_lw, ad_oas).  The softer AD-target mixtures (ad_target_*) blend the
// raw covariance with its projection, so they are NOT orbit-constant and are
// counted as dense estimators for the effective-df below.
bool is_projection_first_ad(const std::string& m) {
  return m.rfind("ad_", 0) == 0 && m.rfind("ad_target", 0) != 0;
}

// Total effective df and the off-diagonal ("edge") count of Sigma under sym.
int model_dim(const EstimatorSpec& spec, const PairSymmetry& sym,
              const Eigen::MatrixXd& Sigma, double tol, int* n_edges) {
  const Eigen::Index p = Sigma.rows();
  int diag = 0, edges = 0;

  if (is_projection_first_ad(spec.method)) {
    // Symmetry-reduced: one parameter per orbit that carries a nonzero value.
    // Orbits are wholly diagonal or wholly off-diagonal (a permutation maps the
    // diagonal to itself), so the first-seen member's kind classifies the orbit.
    std::vector<char> seen(static_cast<std::size_t>(sym.n_orbits), 0);
    std::vector<char> is_diag(static_cast<std::size_t>(sym.n_orbits), 0);
    for (Eigen::Index i = 0; i < p; ++i) {
      for (Eigen::Index j = i; j < p; ++j) {
        if (std::abs(Sigma(i, j)) <= tol) continue;
        const int oid =
            sym.orbit_of[static_cast<std::size_t>(i) * sym.p + static_cast<std::size_t>(j)];
        if (!seen[static_cast<std::size_t>(oid)]) {
          seen[static_cast<std::size_t>(oid)] = 1;
          is_diag[static_cast<std::size_t>(oid)] = (i == j) ? 1 : 0;
        }
      }
    }
    for (int o = 0; o < sym.n_orbits; ++o) {
      if (!seen[static_cast<std::size_t>(o)]) continue;
      if (is_diag[static_cast<std::size_t>(o)]) ++diag; else ++edges;
    }
  } else {
    // Dense: every nonzero upper-triangular entry is its own free parameter.
    for (Eigen::Index i = 0; i < p; ++i)
      for (Eigen::Index j = i; j < p; ++j)
        if (std::abs(Sigma(i, j)) > tol) { if (i == j) ++diag; else ++edges; }
  }
  if (n_edges) *n_edges = edges;
  return diag + edges;
}

// One-pass Extended BIC given data X, a projector, and the symmetry the df is
// counted over.  Fits Sigma once on the full sample, scores the in-sample
// Gaussian deviance, and adds the EBIC penalty.
double ebic_with_projector(const Eigen::MatrixXd& X, const ProjectFn& project,
                           const PairSymmetry& sym, const EstimatorSpec& spec,
                           double gamma) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (n < 2) {
    throw std::invalid_argument("ebic_score: need >= 2 samples");
  }
  const Eigen::MatrixXd Sigma = estimate_with_projector(X, project, spec);
  const Eigen::VectorXd mu = X.colwise().mean().transpose();

  double nll_total = 0.0;  // sum_i -log N(x_i) == -loglik
  for (Eigen::Index i = 0; i < n; ++i) {
    nll_total += gaussian_nll_one(X.row(i).transpose(), mu, Sigma);
  }
  const double neg2ll = 2.0 * nll_total;

  int edges = 0;
  const int dim = model_dim(spec, sym, Sigma, 1e-8, &edges);
  const double dn = static_cast<double>(n);
  const double dp = static_cast<double>(p);
  return neg2ll + static_cast<double>(dim) * std::log(dn) +
         4.0 * gamma * static_cast<double>(edges) * std::log(dp);
}

// k-fold CV mean NLL with contiguous, unshuffled folds (numpy.array_split): the
// first (n mod k) folds hold one extra row.  Refits the estimator once per fold.
double kfold_with_projector(const Eigen::MatrixXd& X, const ProjectFn& project,
                            const EstimatorSpec& spec, int k) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (k < 2) k = 2;
  if (static_cast<Eigen::Index>(k) > n) k = static_cast<int>(n);
  const Eigen::Index base = n / k;
  const Eigen::Index rem = n % k;

  double total = 0.0;
  Eigen::Index start = 0;
  for (int f = 0; f < k; ++f) {
    const Eigen::Index size = base + (static_cast<Eigen::Index>(f) < rem ? 1 : 0);
    if (size == 0) continue;
    const Eigen::Index ntr = n - size;
    if (ntr < 3) {
      throw std::invalid_argument(
          "kfold_nll: training fold has < 3 rows for an unbiased covariance");
    }
    Eigen::MatrixXd Xtr(ntr, p);
    Xtr.topRows(start) = X.topRows(start);
    Xtr.bottomRows(n - (start + size)) = X.bottomRows(n - (start + size));
    const Eigen::VectorXd mu = Xtr.colwise().mean().transpose();
    const Eigen::MatrixXd Sigma = estimate_with_projector(Xtr, project, spec);
    for (Eigen::Index i = start; i < start + size; ++i) {
      total += gaussian_nll_one(X.row(i).transpose(), mu, Sigma);
    }
    start += size;
  }
  return total / static_cast<double>(n);
}

std::vector<EstimatorResult> recommend_with_projector(const Eigen::MatrixXd& X,
                                                      const ProjectFn& project,
                                                      const PairSymmetry& sym,
                                                      const CriterionSpec& crit) {
  const auto grid = candidate_grid(static_cast<int>(X.cols()),
                                   static_cast<int>(X.rows()));
  std::vector<EstimatorResult> results;
  results.reserve(grid.size());

  for (const auto& spec : grid) {
    try {
      double score;
      switch (crit.type) {
        case SelectionCriterion::Ebic:
          score = ebic_with_projector(X, project, sym, spec, crit.gamma);
          break;
        case SelectionCriterion::Kfold:
          score = kfold_with_projector(X, project, spec, crit.k);
          break;
        case SelectionCriterion::Loo:
        default:
          score = loo_with_projector(X, project, spec);
          break;
      }
      const Eigen::MatrixXd Sigma = estimate_with_projector(X, project, spec);
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

// Bind the block-exchangeable projector for a label vector.
ProjectFn block_projector(const std::vector<int>& labels) {
  return [&labels](const Eigen::MatrixXd& S) {
    return reynolds_project(S, labels);
  };
}

// Bind the general-symmetry projector for a pair symmetry.
ProjectFn symmetry_projector(const PairSymmetry& sym) {
  return [&sym](const Eigen::MatrixXd& S) { return reynolds_project(S, sym); };
}

}  // namespace

Eigen::MatrixXd estimate_covariance(const Eigen::MatrixXd& X,
                                    const std::vector<int>& labels,
                                    const EstimatorSpec& spec) {
  if (static_cast<Eigen::Index>(labels.size()) != X.cols()) {
    throw std::invalid_argument(
        "estimate_covariance: labels length must equal number of genes (cols)");
  }
  return estimate_with_projector(X, block_projector(labels), spec);
}

Eigen::MatrixXd estimate_covariance(const Eigen::MatrixXd& X,
                                    const PairSymmetry& sym,
                                    const EstimatorSpec& spec) {
  if (sym.p != static_cast<int>(X.cols())) {
    throw std::invalid_argument(
        "estimate_covariance: sym.p must equal number of genes (cols)");
  }
  return estimate_with_projector(X, symmetry_projector(sym), spec);
}

double gaussian_nll_one(const Eigen::VectorXd& x, const Eigen::VectorXd& mu,
                        const Eigen::MatrixXd& Sigma) {
  const Eigen::Index p = x.size();
  // Every Sigma reaching this function has already been make_pd'd by the
  // estimator dispatch, so the Cholesky succeeds directly in the common case.
  // Attempt LLT first and only fall back to the O(p^3) SPD eigen-projection
  // when the raw matrix is not numerically positive-definite.
  Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
  if (llt.info() != Eigen::Success) {
    llt.compute(make_pd(Sigma));
    if (llt.info() != Eigen::Success) {
      return std::numeric_limits<double>::infinity();
    }
  }
  const Eigen::MatrixXd& L = llt.matrixL();
  double logdet = 0.0;
  for (Eigen::Index i = 0; i < p; ++i) {
    logdet += std::log(L(i, i));
  }
  logdet *= 2.0;
  const Eigen::VectorXd d = x - mu;
  const Eigen::VectorXd y = L.triangularView<Eigen::Lower>().solve(d);
  const double q = y.squaredNorm();
  return 0.5 * (static_cast<double>(p) * std::log(kTwoPi) + logdet + q);
}

double loo_nll(const Eigen::MatrixXd& X, const std::vector<int>& labels,
               const EstimatorSpec& spec) {
  if (static_cast<Eigen::Index>(labels.size()) != X.cols()) {
    throw std::invalid_argument(
        "loo_nll: labels length must equal number of genes (cols)");
  }
  return loo_with_projector(X, block_projector(labels), spec);
}

double loo_nll(const Eigen::MatrixXd& X, const PairSymmetry& sym,
               const EstimatorSpec& spec) {
  if (sym.p != static_cast<int>(X.cols())) {
    throw std::invalid_argument(
        "loo_nll: sym.p must equal number of genes (cols)");
  }
  return loo_with_projector(X, symmetry_projector(sym), spec);
}

int effective_df(const EstimatorSpec& spec, const PairSymmetry& sym,
                 const Eigen::MatrixXd& Sigma, double tol, int* n_edges) {
  if (sym.p != static_cast<int>(Sigma.rows())) {
    throw std::invalid_argument(
        "effective_df: sym.p must equal the covariance dimension");
  }
  return model_dim(spec, sym, Sigma, tol, n_edges);
}

double ebic_score(const Eigen::MatrixXd& X, const std::vector<int>& labels,
                  const EstimatorSpec& spec, double gamma) {
  if (static_cast<Eigen::Index>(labels.size()) != X.cols()) {
    throw std::invalid_argument(
        "ebic_score: labels length must equal number of genes (cols)");
  }
  const PairSymmetry sym = pair_symmetry_from_labels(labels);
  return ebic_with_projector(X, block_projector(labels), sym, spec, gamma);
}

double ebic_score(const Eigen::MatrixXd& X, const PairSymmetry& sym,
                  const EstimatorSpec& spec, double gamma) {
  if (sym.p != static_cast<int>(X.cols())) {
    throw std::invalid_argument(
        "ebic_score: sym.p must equal number of genes (cols)");
  }
  return ebic_with_projector(X, symmetry_projector(sym), sym, spec, gamma);
}

double kfold_nll(const Eigen::MatrixXd& X, const std::vector<int>& labels,
                 const EstimatorSpec& spec, int k) {
  if (static_cast<Eigen::Index>(labels.size()) != X.cols()) {
    throw std::invalid_argument(
        "kfold_nll: labels length must equal number of genes (cols)");
  }
  return kfold_with_projector(X, block_projector(labels), spec, k);
}

double kfold_nll(const Eigen::MatrixXd& X, const PairSymmetry& sym,
                 const EstimatorSpec& spec, int k) {
  if (sym.p != static_cast<int>(X.cols())) {
    throw std::invalid_argument(
        "kfold_nll: sym.p must equal number of genes (cols)");
  }
  return kfold_with_projector(X, symmetry_projector(sym), spec, k);
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
  // Symmetry-target ("AD-target") family: shrink toward P_G(S) instead of
  // projecting onto it.  Added as extra candidates; leave-one-out NLL selects
  // them only when the softer prior genuinely fits better, so the recommendation
  // can only improve or tie relative to the projection-first grid.
  grid.push_back({"ad_target_lw", {}});
  grid.push_back({"ad_target_oas", {}});
  for (double lam : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    grid.push_back({"ad_target_ridge", {{"lam", lam}}});
  }
  return grid;
}

std::vector<EstimatorResult> recommend_estimator(
    const Eigen::MatrixXd& X, const std::vector<int>& labels,
    const CriterionSpec& crit) {
  if (static_cast<Eigen::Index>(labels.size()) != X.cols()) {
    throw std::invalid_argument(
        "recommend_estimator: labels length must equal number of genes (cols)");
  }
  const PairSymmetry sym = pair_symmetry_from_labels(labels);
  return recommend_with_projector(X, block_projector(labels), sym, crit);
}

std::vector<EstimatorResult> recommend_estimator(const Eigen::MatrixXd& X,
                                                 const PairSymmetry& sym,
                                                 const CriterionSpec& crit) {
  if (sym.p != static_cast<int>(X.cols())) {
    throw std::invalid_argument(
        "recommend_estimator: sym.p must equal number of genes (cols)");
  }
  return recommend_with_projector(X, symmetry_projector(sym), sym, crit);
}

std::vector<EstimatorResult> recommend_estimator(
    const Eigen::MatrixXd& X, const std::vector<int>& labels) {
  return recommend_estimator(X, labels, CriterionSpec{});
}

std::vector<EstimatorResult> recommend_estimator(const Eigen::MatrixXd& X,
                                                 const PairSymmetry& sym) {
  return recommend_estimator(X, sym, CriterionSpec{});
}

}  // namespace adgencov
