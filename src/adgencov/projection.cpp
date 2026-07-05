#include "adgencov/projection.hpp"

#include <array>
#include <map>
#include <numeric>
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

// ---------------------------------------------------------------------------
// General orbit-averaging (commutant) projection.
// ---------------------------------------------------------------------------

namespace {

// Dense-relabel a per-pair root array into contiguous orbit ids [0, k),
// assigned in first-appearance (row-major) order for deterministic output.
int densify(std::vector<int>& orbit_of) {
  std::unordered_map<int, int> id;
  id.reserve(orbit_of.size());
  int next = 0;
  for (int& v : orbit_of) {
    auto it = id.find(v);
    if (it == id.end()) {
      id.emplace(v, next);
      v = next;
      ++next;
    } else {
      v = it->second;
    }
  }
  return next;
}

// Minimal union-find over N elements with path halving + union by size.
struct UnionFind {
  std::vector<int> parent, size;
  explicit UnionFind(int n) : parent(n), size(n, 1) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (size[a] < size[b]) std::swap(a, b);
    parent[b] = a;
    size[a] += size[b];
  }
};

void validate_permutation(int p, const std::vector<int>& g) {
  if (static_cast<int>(g.size()) != p)
    throw std::invalid_argument(
        "pair_symmetry_from_generators: generator length must equal p");
  std::vector<char> seen(p, 0);
  for (int v : g) {
    if (v < 0 || v >= p)
      throw std::invalid_argument(
          "pair_symmetry_from_generators: generator entry out of range [0,p)");
    if (seen[v])
      throw std::invalid_argument(
          "pair_symmetry_from_generators: generator is not a permutation");
    seen[v] = 1;
  }
}

}  // namespace

PairSymmetry pair_symmetry_from_labels(const std::vector<int>& labels) {
  const int p = static_cast<int>(labels.size());
  if (p < 1)
    throw std::invalid_argument("pair_symmetry_from_labels: labels is empty");

  // Dense block id per gene, first-appearance order (matches reynolds_project).
  std::unordered_map<int, int> block_of;
  std::vector<int> blk(p);
  for (int i = 0; i < p; ++i) {
    auto it = block_of.find(labels[i]);
    if (it == block_of.end()) {
      const int b = static_cast<int>(block_of.size());
      block_of.emplace(labels[i], b);
      blk[i] = b;
    } else {
      blk[i] = it->second;
    }
  }

  // Orbit key: kind 0 = within-block diagonal, 1 = within-block off-diagonal,
  // 2 = ordered cross-block rectangle.  Same three orbit kinds the block
  // projection averages over, so the projected values coincide.
  std::map<std::array<int, 3>, int> key_id;
  PairSymmetry sym;
  sym.p = p;
  sym.orbit_of.resize(static_cast<std::size_t>(p) * p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      std::array<int, 3> k;
      if (blk[i] == blk[j])
        k = (i == j) ? std::array<int, 3>{0, blk[i], blk[i]}
                     : std::array<int, 3>{1, blk[i], blk[i]};
      else
        k = std::array<int, 3>{2, blk[i], blk[j]};
      auto it = key_id.find(k);
      int id;
      if (it == key_id.end()) {
        id = static_cast<int>(key_id.size());
        key_id.emplace(k, id);
      } else {
        id = it->second;
      }
      sym.orbit_of[static_cast<std::size_t>(i) * p + j] = id;
    }
  }
  sym.n_orbits = static_cast<int>(key_id.size());
  return sym;
}

PairSymmetry pair_symmetry_from_generators(
    int p, const std::vector<std::vector<int>>& generators) {
  if (p < 1)
    throw std::invalid_argument("pair_symmetry_from_generators: p must be >= 1");
  for (const auto& g : generators) validate_permutation(p, g);

  const int N = p * p;
  UnionFind uf(N);
  // Flood-fill: for each generator g, (i,j) ~ (g[i], g[j]).  Closing under the
  // generators is sufficient — the group they generate has the same pair orbits.
  for (const auto& g : generators) {
    for (int i = 0; i < p; ++i) {
      for (int j = 0; j < p; ++j) {
        const int a = i * p + j;
        const int b = g[i] * p + g[j];
        uf.unite(a, b);
      }
    }
  }

  PairSymmetry sym;
  sym.p = p;
  sym.orbit_of.resize(static_cast<std::size_t>(N));
  for (int a = 0; a < N; ++a) sym.orbit_of[a] = uf.find(a);
  sym.n_orbits = densify(sym.orbit_of);
  return sym;
}

PairSymmetry pair_symmetry_banded(int p) {
  if (p < 1)
    throw std::invalid_argument("pair_symmetry_banded: p must be >= 1");
  PairSymmetry sym;
  sym.p = p;
  sym.orbit_of.resize(static_cast<std::size_t>(p) * p);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j < p; ++j)
      sym.orbit_of[static_cast<std::size_t>(i) * p + j] = (i >= j) ? (i - j) : (j - i);
  sym.n_orbits = p;  // bands 0..p-1
  return sym;
}

Eigen::MatrixXd reynolds_project(const Eigen::MatrixXd& S,
                                 const PairSymmetry& sym) {
  const Eigen::Index p = S.rows();
  if (S.cols() != p)
    throw std::invalid_argument("reynolds_project: S must be square");
  if (sym.p != static_cast<int>(p))
    throw std::invalid_argument(
        "reynolds_project: sym.p must equal S.rows()");
  if (static_cast<Eigen::Index>(sym.orbit_of.size()) != p * p)
    throw std::invalid_argument(
        "reynolds_project: sym.orbit_of has the wrong size");

  std::vector<double> sum(static_cast<std::size_t>(sym.n_orbits), 0.0);
  std::vector<double> cnt(static_cast<std::size_t>(sym.n_orbits), 0.0);
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = 0; j < p; ++j) {
      const int o = sym.orbit_of[static_cast<std::size_t>(i) * p + j];
      sum[static_cast<std::size_t>(o)] += S(i, j);
      cnt[static_cast<std::size_t>(o)] += 1.0;
    }
  }
  Eigen::MatrixXd P(p, p);
  for (Eigen::Index i = 0; i < p; ++i) {
    for (Eigen::Index j = 0; j < p; ++j) {
      const int o = sym.orbit_of[static_cast<std::size_t>(i) * p + j];
      P(i, j) = sum[static_cast<std::size_t>(o)] / cnt[static_cast<std::size_t>(o)];
    }
  }
  // Symmetric already for a valid pair symmetry; enforce exactly for safety.
  P = 0.5 * (P + P.transpose()).eval();
  return P;
}

}  // namespace adgencov
