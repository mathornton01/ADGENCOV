#include "adgencov/clustering.hpp"

#include <limits>
#include <stdexcept>

namespace adgencov {

std::vector<int> agglomerative_average(const Eigen::MatrixXd& dist, int n_clusters) {
  const int n = static_cast<int>(dist.rows());
  if (dist.cols() != n)
    throw std::invalid_argument("adgencov: distance matrix must be square.");
  if (n_clusters < 1 || n_clusters > n)
    throw std::invalid_argument("adgencov: n_clusters must be in [1, n_samples].");

  // Working copy of the pairwise distances; rows/cols for merged-away clusters
  // are simply marked inactive rather than physically removed.
  Eigen::MatrixXd d = dist;
  std::vector<bool> active(n, true);
  std::vector<int> csize(n, 1);
  // Original-sample members of each current cluster (indexed by the surviving
  // representative row).
  std::vector<std::vector<int>> members(n);
  for (int i = 0; i < n; ++i) members[i] = {i};

  int nclust = n;
  while (nclust > n_clusters) {
    // Find the closest active pair (i < j).  Ties break toward the smallest
    // (i, j) in row-major order, matching a deterministic scan.
    double best = std::numeric_limits<double>::infinity();
    int bi = -1, bj = -1;
    for (int i = 0; i < n; ++i) {
      if (!active[i]) continue;
      for (int j = i + 1; j < n; ++j) {
        if (!active[j]) continue;
        if (d(i, j) < best) {
          best = d(i, j);
          bi = i;
          bj = j;
        }
      }
    }

    // Merge bj into bi with the size-weighted (average-linkage) update.
    const double wi = static_cast<double>(csize[bi]);
    const double wj = static_cast<double>(csize[bj]);
    for (int k = 0; k < n; ++k) {
      if (!active[k] || k == bi || k == bj) continue;
      const double nd = (wi * d(bi, k) + wj * d(bj, k)) / (wi + wj);
      d(bi, k) = nd;
      d(k, bi) = nd;
    }
    csize[bi] += csize[bj];
    members[bi].insert(members[bi].end(), members[bj].begin(), members[bj].end());
    active[bj] = false;
    --nclust;
  }

  // Assign cluster ids in ascending order of the surviving representative row,
  // which (since members always retain their original rows) is ascending order
  // of each cluster's smallest member index.
  std::vector<int> labels(n, -1);
  int cid = 0;
  for (int i = 0; i < n; ++i) {
    if (!active[i]) continue;
    for (int m : members[i]) labels[m] = cid;
    ++cid;
  }
  return labels;
}

}  // namespace adgencov
