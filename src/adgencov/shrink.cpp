#include "adgencov/shrink.hpp"

#include <algorithm>
#include <stdexcept>

namespace adgencov {

namespace {

// Column-center X (subtract each column's mean), matching numpy X - X.mean(0).
Eigen::MatrixXd center_columns(const Eigen::MatrixXd& X) {
  const Eigen::RowVectorXd mean = X.colwise().mean();
  return X.rowwise() - mean;
}

}  // namespace

Eigen::MatrixXd sample_covariance(const Eigen::MatrixXd& X, bool unbiased) {
  const Eigen::Index n = X.rows();
  if (n < 1) {
    throw std::invalid_argument("sample_covariance: X has no rows");
  }
  if (unbiased && n < 2) {
    throw std::invalid_argument(
        "sample_covariance: unbiased estimate needs >= 2 samples");
  }
  const Eigen::MatrixXd Xc = center_columns(X);
  const double denom = unbiased ? static_cast<double>(n - 1)
                                : static_cast<double>(n);
  Eigen::MatrixXd S = (Xc.transpose() * Xc) / denom;
  // Enforce exact symmetry (guards against round-off asymmetry).
  return (0.5 * (S + S.transpose())).eval();
}

Eigen::MatrixXd make_pd(const Eigen::MatrixXd& S, double floor) {
  const Eigen::MatrixXd Sym = 0.5 * (S + S.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sym);
  if (es.info() != Eigen::Success) {
    throw std::runtime_error("make_pd: eigen-decomposition failed");
  }
  Eigen::VectorXd vals = es.eigenvalues();
  for (Eigen::Index i = 0; i < vals.size(); ++i) {
    if (vals[i] < floor) vals[i] = floor;
  }
  const Eigen::MatrixXd& V = es.eigenvectors();
  return V * vals.asDiagonal() * V.transpose();
}

Eigen::MatrixXd ridge(const Eigen::MatrixXd& S, double alpha) {
  const Eigen::Index p = S.rows();
  if (S.cols() != p) throw std::invalid_argument("ridge: S must be square");
  const double mu = S.trace() / static_cast<double>(p);
  Eigen::MatrixXd R = (1.0 - alpha) * S;
  R.diagonal().array() += alpha * mu;
  return R;
}

Eigen::MatrixXd shrink_to_target(const Eigen::MatrixXd& S,
                                 const Eigen::MatrixXd& target, double lambda) {
  if (S.rows() != target.rows() || S.cols() != target.cols())
    throw std::invalid_argument("shrink_to_target: S and target sizes disagree");
  const double l = std::max(0.0, std::min(1.0, lambda));
  return (1.0 - l) * S + l * target;
}

Eigen::MatrixXd soft_threshold_offdiag(const Eigen::MatrixXd& S, double lam,
                                       double l1_ratio) {
  const Eigen::Index p = S.rows();
  if (S.cols() != p) {
    throw std::invalid_argument("soft_threshold_offdiag: S must be square");
  }
  const double thresh = lam * l1_ratio;
  const double ridge_add = lam * (1.0 - l1_ratio);
  Eigen::MatrixXd R(p, p);
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = 0; j < p; ++j) {
      if (i == j) {
        R(i, j) = S(i, j);  // diagonal preserved; ridge term added below
      } else {
        const double s = S(i, j);
        const double mag = std::max(std::abs(s) - thresh, 0.0);
        R(i, j) = (s > 0.0 ? 1.0 : (s < 0.0 ? -1.0 : 0.0)) * mag;
      }
    }
  }
  R.diagonal().array() += ridge_add;
  return R;
}

double ledoit_wolf_shrinkage(const Eigen::MatrixXd& X) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (n < 1 || p < 1) {
    throw std::invalid_argument("ledoit_wolf_shrinkage: empty X");
  }
  const double dn = static_cast<double>(n);
  const double dp = static_cast<double>(p);

  const Eigen::MatrixXd Xc = center_columns(X);
  const Eigen::MatrixXd X2 = Xc.array().square().matrix();  // elementwise square

  // emp_cov = Xc^T Xc / n  (biased);  mu = tr(emp_cov)/p.
  const Eigen::MatrixXd gram = Xc.transpose() * Xc;  // = n * emp_cov
  const double mu = (gram.trace() / dn) / dp;

  // beta_  = sum of all entries of (X2^T X2)
  // delta_ = ||emp_cov||_F^2 = sum((gram)^2) / n^2
  const double beta_raw = (X2.transpose() * X2).sum();
  const double delta_ = gram.array().square().sum() / (dn * dn);

  double beta = (1.0 / (dp * dn)) * (beta_raw / dn - delta_);
  double delta = (delta_ - dp * mu * mu) / dp;

  beta = std::min(beta, delta);
  if (delta == 0.0 || beta == 0.0) return 0.0;
  double shrinkage = beta / delta;
  // Clamp for numerical safety (theory guarantees [0,1]).
  return std::min(std::max(shrinkage, 0.0), 1.0);
}

Eigen::MatrixXd ledoit_wolf(const Eigen::MatrixXd& X) {
  const Eigen::MatrixXd emp = sample_covariance(X, /*unbiased=*/false);
  const double d = ledoit_wolf_shrinkage(X);
  const double mu = emp.trace() / static_cast<double>(emp.rows());
  Eigen::MatrixXd cov = (1.0 - d) * emp;
  cov.diagonal().array() += d * mu;
  return cov;
}

double oas_shrinkage(const Eigen::MatrixXd& X) {
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  if (n < 1 || p < 1) throw std::invalid_argument("oas_shrinkage: empty X");
  const double dn = static_cast<double>(n);
  const double dp = static_cast<double>(p);

  const Eigen::MatrixXd emp = sample_covariance(X, /*unbiased=*/false);
  const double mu = emp.trace() / dp;
  const double alpha = emp.array().square().mean();  // mean(emp_cov^2)

  const double num = alpha + mu * mu;
  const double den = (dn + 1.0) * (alpha - (mu * mu) / dp);
  if (den == 0.0) return 1.0;
  return std::min(num / den, 1.0);
}

Eigen::MatrixXd oas(const Eigen::MatrixXd& X) {
  const Eigen::MatrixXd emp = sample_covariance(X, /*unbiased=*/false);
  const double d = oas_shrinkage(X);
  const double mu = emp.trace() / static_cast<double>(emp.rows());
  Eigen::MatrixXd cov = (1.0 - d) * emp;
  cov.diagonal().array() += d * mu;
  return cov;
}

}  // namespace adgencov
