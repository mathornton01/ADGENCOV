#include "adgencov/projection.hpp"

#include <stdexcept>
#include <unordered_map>

namespace adgencov {

Eigen::MatrixXd reynolds_project(const Eigen::MatrixXd& S,
                                 const std::vector<int>& labels) {
  const Eigen::Index p = S.rows();
  if (S.cols() != p) {
    throw std::invalid_argument("reynolds_project: S must be square");
  }
  if (static_cast<Eigen::Index>(labels.size()) != p) {
    throw std::invalid_argument(
        "reynolds_project: labels.size() must equal S.rows()");
  }

  // Map arbitrary integer labels -> contiguous block ids [0, g), preserving
  // first-appearance order, and collect the member indices of each block.
  std::unordered_map<int, int> block_of;
  std::vector<std::vector<Eigen::Index>> members;
  members.reserve(static_cast<std::size_t>(p));
  for (Eigen::Index i = 0; i < p; ++i) {
    auto it = block_of.find(labels[i]);
    int b;
    if (it == block_of.end()) {
      b = static_cast<int>(members.size());
      block_of.emplace(labels[i], b);
      members.emplace_back();
    } else {
      b = it->second;
    }
    members[b].push_back(i);
  }
  const std::size_t g = members.size();

  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(p, p);

  // (i) + (ii): within-block averaging of diagonal and off-diagonal entries.
  for (std::size_t b = 0; b < g; ++b) {
    const std::vector<Eigen::Index>& idx = members[b];
    const std::size_t n = idx.size();
    if (n == 1) {
      const Eigen::Index i = idx[0];
      P(i, i) = S(i, i);
      continue;
    }
    double diag_sum = 0.0;
    double off_sum = 0.0;
    for (std::size_t a = 0; a < n; ++a) {
      diag_sum += S(idx[a], idx[a]);
      for (std::size_t c = 0; c < n; ++c) {
        if (a != c) off_sum += S(idx[a], idx[c]);
      }
    }
    const double diag_mean = diag_sum / static_cast<double>(n);
    const double off_mean = off_sum / static_cast<double>(n * (n - 1));
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t c = 0; c < n; ++c) {
        P(idx[a], idx[c]) = (a == c) ? diag_mean : off_mean;
      }
    }
  }

  // (iii): cross-block averaging for each unordered pair of blocks.
  for (std::size_t b = 0; b < g; ++b) {
    const std::vector<Eigen::Index>& ia = members[b];
    for (std::size_t d = b + 1; d < g; ++d) {
      const std::vector<Eigen::Index>& ib = members[d];
      double sum = 0.0;
      for (const Eigen::Index i : ia) {
        for (const Eigen::Index j : ib) sum += S(i, j);
      }
      const double val = sum / static_cast<double>(ia.size() * ib.size());
      for (const Eigen::Index i : ia) {
        for (const Eigen::Index j : ib) {
          P(i, j) = val;
          P(j, i) = val;
        }
      }
    }
  }

  // Enforce exact numerical symmetry.
  P = 0.5 * (P + P.transpose()).eval();
  return P;
}

}  // namespace adgencov
