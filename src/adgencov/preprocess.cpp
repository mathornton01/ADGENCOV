#include "adgencov/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace adgencov {

namespace {
constexpr double EPS = 1e-7;
}

Dataset preprocess(const ExpressionData& data, int n_genes, double min_mean,
                   bool log_transform) {
  const int ng = static_cast<int>(data.genes.size());
  const int ns = static_cast<int>(data.values.cols());

  // --- Step 1: fillna(0) + clip(lower=0), and per-gene mean abundance -------
  Eigen::MatrixXd V = data.values;  // genes x samples
  for (int i = 0; i < ng; ++i)
    for (int j = 0; j < ns; ++j) {
      double v = V(i, j);
      if (!std::isfinite(v) || v < 0.0) v = 0.0;
      V(i, j) = v;
    }
  Eigen::VectorXd row_mean = V.rowwise().mean();

  // --- Step 2: collapse duplicate symbols, keeping the largest-mean row -----
  // Emulate sort_values("_mean", desc).drop_duplicates("gene", keep="first").
  std::unordered_map<std::string, int> best;  // gene -> row index kept
  for (int i = 0; i < ng; ++i) {
    auto it = best.find(data.genes[i]);
    if (it == best.end() || row_mean(i) > row_mean(it->second))
      best[data.genes[i]] = i;
  }

  // --- Step 3: filter by min_mean; collect survivors ------------------------
  std::vector<int> keep;
  keep.reserve(best.size());
  for (const auto& kv : best)
    if (row_mean(kv.second) >= min_mean) keep.push_back(kv.second);

  // Order survivors by descending mean (pandas' post-sort order), tie-break by
  // original row index ascending for determinism.
  std::sort(keep.begin(), keep.end(), [&](int a, int b) {
    if (row_mean(a) != row_mean(b)) return row_mean(a) > row_mean(b);
    return a < b;
  });

  const int m = static_cast<int>(keep.size());
  Eigen::MatrixXd Xg(m, ns);  // genes x samples, ordered
  for (int r = 0; r < m; ++r) Xg.row(r) = V.row(keep[r]);

  // --- Step 4: optional log2(x + 1) -----------------------------------------
  if (log_transform)
    Xg = (Xg.array() + 1.0).log() / std::log(2.0);

  // --- Step 5: keep highest-variance genes (population variance, ddof=0) ----
  Eigen::VectorXd var(m);
  for (int r = 0; r < m; ++r) {
    Eigen::RowVectorXd row = Xg.row(r);
    double mu = row.mean();
    var(r) = (row.array() - mu).square().sum() / static_cast<double>(ns);
  }
  std::vector<int> order(m);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (var(a) != var(b)) return var(a) > var(b);
    return a < b;
  });
  const int take = std::min(n_genes, m);

  // --- Step 6: transpose, center, z-score (ddof=1) --------------------------
  Dataset out;
  out.genes.resize(take);
  Eigen::MatrixXd X(ns, take);  // samples x genes
  for (int c = 0; c < take; ++c) {
    const int r = order[c];
    out.genes[c] = data.genes[keep[r]];
    X.col(c) = Xg.row(r).transpose();
  }
  for (int c = 0; c < take; ++c) {
    double mu = X.col(c).mean();
    X.col(c).array() -= mu;
    double ss = X.col(c).squaredNorm();
    double sd = (ns > 1) ? std::sqrt(ss / static_cast<double>(ns - 1)) : 0.0;
    if (sd < EPS) sd = 1.0;
    X.col(c).array() /= sd;
  }
  out.X = std::move(X);
  return out;
}

}  // namespace adgencov
