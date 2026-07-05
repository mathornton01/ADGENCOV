#ifndef ADGENCOV_SELECT_HPP
#define ADGENCOV_SELECT_HPP

#include <map>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "adgencov/projection.hpp"  // PairSymmetry, reynolds_project

/// @file select.hpp
/// Estimator recommender and Gaussian likelihood for ADGENCOV.
///
/// This is the model-selection layer of the Applications Note prototype,
/// ported to C++17/Eigen.  It provides:
///   * @c estimate_covariance — the method dispatcher that maps a method name +
///     parameters onto the estimator family in shrink.hpp (with AD variants
///     applying the Reynolds projection from projection.hpp first);
///   * @c gaussian_nll_one   — the per-sample multivariate-Gaussian negative
///     log-likelihood used as the leave-one-out scoring rule;
///   * @c loo_nll            — the leave-one-out cross-validated NLL, computed
///     with an exact rank-1 covariance downdate so each fold is O(p^2) instead
///     of recomputing the full scatter matrix;
///   * @c candidate_grid     — the conservative estimator grid;
///   * @c recommend_estimator — the ranked recommendation over the grid.
///
/// All numerics mirror ad_covariance_app.py exactly and are validated to ~1e-9
/// against golden values in tests/gen_golden.py.

namespace adgencov {

/// A candidate estimator: a method name plus its scalar hyper-parameters.
///
/// Recognised method names (matching the prototype's estimate_covariance):
///   sample, ad_sample, ridge, ad_ridge, lasso, ad_lasso, elastic_net,
///   ad_elastic_net, lw / ledoit_wolf, oas, ad_linear_lw, ad_oas.
///   Symmetry-target ("AD-target") variants, which shrink the raw covariance
///   *toward* the projected estimate P_G(S) instead of projecting onto it hard:
///   ad_target_ridge, ad_target_lw, ad_target_oas.
/// Recognised parameter keys: "alpha" (ridge, default 0.2), "lam" (sparse,
/// default 0.05; also the AD-target strength for ad_target_ridge, default 0.5),
/// "l1_ratio" (elastic-net, default 0.25 in the grid), "diag_alpha" (optional
/// identity ridge for the AD-target family; default 1e-3 for ad_target_ridge,
/// 0 for ad_target_lw/oas).
struct EstimatorSpec {
  std::string method;
  std::map<std::string, double> params;
};

/// Result of scoring one candidate on the full data set.
struct EstimatorResult {
  EstimatorSpec spec;             ///< method + parameters
  Eigen::MatrixXd covariance;     ///< full-data estimate (make_pd'd, SPD)
  double loo_nll;                 ///< leave-one-out cross-validated NLL (lower is better)
  double condition_number;        ///< 2-norm condition number of @c covariance
};

/// Dispatch to the estimator named by @c spec on samples-by-genes matrix @c X
/// (n rows, p cols) with block @c labels (length p).  AD variants apply the
/// Reynolds projection to the unbiased sample covariance first.  The returned
/// matrix is always passed through make_pd (SPD, eigenvalues floored at 1e-5),
/// exactly as the prototype's estimate_covariance does.
/// @throws std::invalid_argument for an unknown method or a labels/X mismatch.
Eigen::MatrixXd estimate_covariance(const Eigen::MatrixXd& X,
                                    const std::vector<int>& labels,
                                    const EstimatorSpec& spec);

/// General-symmetry overload: identical to the labels form, but AD variants
/// project through the arbitrary group commutant @c sym (from
/// pair_symmetry_from_generators / _banded / _from_labels) instead of the
/// block-exchangeable projection.  With @c sym = pair_symmetry_from_labels(l)
/// this reproduces the labels overload (up to summation order).
/// @throws std::invalid_argument for an unknown method or @c sym.p != X.cols().
Eigen::MatrixXd estimate_covariance(const Eigen::MatrixXd& X,
                                    const PairSymmetry& sym,
                                    const EstimatorSpec& spec);

/// Negative log-likelihood of one observation @c x under N(@c mu, @c Sigma):
///   0.5 * (p*log(2*pi) + logdet(Sigma) + (x-mu)^T Sigma^{-1} (x-mu)).
/// @c Sigma is passed through make_pd first (matching the prototype); if it is
/// still non-positive-definite the function returns +infinity.
double gaussian_nll_one(const Eigen::VectorXd& x, const Eigen::VectorXd& mu,
                        const Eigen::MatrixXd& Sigma);

/// Leave-one-out cross-validated mean negative log-likelihood of @c spec on
/// @c X.  For each held-out row i the training mean and covariance are formed
/// from the remaining n-1 rows and scored on row i; the mean over folds is
/// returned.
///
/// For covariance-only methods (sample/ridge/lasso/elastic-net and their AD
/// variants) the per-fold sample covariance is obtained from the full scatter
/// matrix by an exact rank-1 downdate, so no per-fold submatrix is built.  For
/// the data-driven Ledoit-Wolf / OAS methods the training submatrix is formed
/// because their shrinkage intensities depend on X directly.
double loo_nll(const Eigen::MatrixXd& X, const std::vector<int>& labels,
               const EstimatorSpec& spec);

/// General-symmetry overload of @c loo_nll (AD variants project through @c sym).
double loo_nll(const Eigen::MatrixXd& X, const PairSymmetry& sym,
               const EstimatorSpec& spec);

/// The conservative candidate grid used by @c recommend_estimator, depending on
/// the problem shape (p genes, n samples).  Mirrors the prototype's
/// candidate_grid: a ridge sweep, the four data-driven shrinkers, and a
/// LASSO / elastic-net lambda sweep.
std::vector<EstimatorSpec> candidate_grid(int p, int n);

/// Score every candidate from @c candidate_grid on @c X and return the results
/// sorted by ascending leave-one-out NLL (best first).  Candidates whose
/// estimation throws are skipped (matching the prototype's try/except).
std::vector<EstimatorResult> recommend_estimator(const Eigen::MatrixXd& X,
                                                 const std::vector<int>& labels);

/// General-symmetry overload of @c recommend_estimator: score the candidate
/// grid with AD variants projecting through the arbitrary group commutant
/// @c sym, sorted ascending by leave-one-out NLL.
std::vector<EstimatorResult> recommend_estimator(const Eigen::MatrixXd& X,
                                                 const PairSymmetry& sym);

}  // namespace adgencov

#endif  // ADGENCOV_SELECT_HPP
