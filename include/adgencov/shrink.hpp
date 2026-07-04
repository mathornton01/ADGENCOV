#ifndef ADGENCOV_SHRINK_HPP
#define ADGENCOV_SHRINK_HPP

#include <vector>
#include <Eigen/Dense>

/// @file shrink.hpp
/// Covariance estimators for ADGENCOV.
///
/// The estimator family mirrors the AD-Covariance Applications Note prototype
/// but in vectorised, allocation-lean C++17/Eigen.  Two sample-covariance
/// conventions appear and are documented per-function:
///   * "unbiased"  divides by (n - 1)  — used by the linear/sparse estimators,
///                                        matching numpy.cov's default.
///   * "biased"    divides by  n        — used internally by Ledoit-Wolf and
///                                        OAS, matching scikit-learn.
///
/// Every "ad_*" variant first applies the Reynolds (commutant) projection from
/// projection.hpp to the sample covariance, then applies the corresponding
/// linear estimator.  This is exactly what the Note's estimate_covariance does.

namespace adgencov {

/// Sample covariance of a samples-by-genes matrix @c X (n rows, p cols).
/// @param unbiased  divide by (n-1) when true (default), else by n.
/// @throws std::invalid_argument if n < 2 with unbiased=true.
Eigen::MatrixXd sample_covariance(const Eigen::MatrixXd& X, bool unbiased = true);

/// Project @c S onto the nearest symmetric positive-definite matrix by flooring
/// eigenvalues at @c floor (eigen-decomposition; symmetrised first).
Eigen::MatrixXd make_pd(const Eigen::MatrixXd& S, double floor = 1e-5);

/// Ridge / diagonal-loading shrinkage toward a scaled identity:
///   (1 - alpha) * S + alpha * (tr(S)/p) * I.
/// @param alpha  shrinkage intensity in [0, 1].
Eigen::MatrixXd ridge(const Eigen::MatrixXd& S, double alpha);

/// Soft-threshold the off-diagonal entries (LASSO / elastic-net covariance):
///   off_ij <- sign(S_ij) * max(|S_ij| - lam * l1_ratio, 0)
/// with an additional ridge term lam * (1 - l1_ratio) added to the diagonal.
/// Diagonal entries are preserved (plus the ridge term).  l1_ratio = 1 gives
/// pure LASSO; l1_ratio < 1 gives elastic-net.
Eigen::MatrixXd soft_threshold_offdiag(const Eigen::MatrixXd& S, double lam,
                                       double l1_ratio = 1.0);

/// Ledoit-Wolf optimal shrinkage intensity toward mu*I (mu = tr/p), computed
/// from the data @c X (n-by-p).  Matches scikit-learn's ledoit_wolf_shrinkage
/// (uses the biased empirical covariance internally).  Returns a value in
/// [0, 1].
double ledoit_wolf_shrinkage(const Eigen::MatrixXd& X);

/// Ledoit-Wolf covariance estimate: (1 - d) * emp_cov + d * mu * I where
/// emp_cov is the biased sample covariance and d = ledoit_wolf_shrinkage(X).
Eigen::MatrixXd ledoit_wolf(const Eigen::MatrixXd& X);

/// Oracle Approximating Shrinkage intensity (Chen et al. 2010), matching
/// scikit-learn's OAS.  Returns a value in [0, 1].
double oas_shrinkage(const Eigen::MatrixXd& X);

/// OAS covariance estimate: (1 - d) * emp_cov + d * mu * I with d = oas_shrinkage(X).
Eigen::MatrixXd oas(const Eigen::MatrixXd& X);

}  // namespace adgencov

#endif  // ADGENCOV_SHRINK_HPP
