#include "adgencov/groups.hpp"

#include <array>
#include <cctype>
#include <cmath>
#include <regex>
#include <stdexcept>
#include <unordered_map>

#include "adgencov/clustering.hpp"

namespace adgencov {

std::string gene_family_label(const std::string& gene) {
  std::string g = gene;
  for (char& c : g) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  static const std::array<const char*, 15> prefixes = {
      "MIR", "SNORD", "SNORA", "COL", "ABCA", "ABCB", "ABCC", "ACAD",
      "RPS", "RPL", "HIST", "KRT", "IGF", "HLA"};
  for (const char* p : prefixes) {
    if (!p) continue;
    const std::string pref(p);
    if (g.size() >= pref.size() && g.compare(0, pref.size(), pref) == 0) return pref;
  }
  // Leading alphabetic run, truncated to 4 characters.
  std::string alpha;
  for (char c : g) {
    if (std::isalpha(static_cast<unsigned char>(c))) alpha.push_back(c);
    else break;
  }
  if (alpha.empty()) return "OTHER";
  return alpha.substr(0, 4);
}

namespace {

// Build gene->value map from a table with a "gene" column and one target
// column, matching prototype behaviour (lowercased headers, first wins).
std::unordered_map<std::string, std::string> table_map(const Table& t,
                                                       const std::string& target) {
  int gcol = -1, tcol = -1;
  for (int c = 0; c < t.ncol(); ++c) {
    std::string h = t.headers[c];
    for (char& ch : h) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (h == "gene") gcol = c;
    else if (h == target) tcol = c;
  }
  if (gcol < 0 || tcol < 0)
    throw std::invalid_argument("adgencov: table must contain columns gene and " + target + ".");
  std::unordered_map<std::string, std::string> m;
  for (const auto& r : t.rows) {
    const std::string& g = r[gcol];
    const std::string& v = r[tcol];
    if (g.empty() || v.empty()) continue;  // dropna
    m.emplace(g, v);                        // first occurrence wins
  }
  return m;
}

// Correlation-distance matrix D = 1 - |corr(X, columns)|, matching the
// prototype: C = np.corrcoef(X, rowvar=False); C = nan_to_num(C, 0); D = 1-|C|.
// A gene with zero variance yields NaN correlations -> treated as 0 (distance
// 1), exactly like numpy's nan_to_num.
Eigen::MatrixXd correlation_distance(const Eigen::MatrixXd& X) {
  const int p = static_cast<int>(X.cols());
  // Center each column, then form the (unscaled) covariance via a Gram matrix.
  Eigen::MatrixXd Xc = X.rowwise() - X.colwise().mean();
  Eigen::MatrixXd cov = Xc.transpose() * Xc;  // proportional to the covariance
  Eigen::VectorXd sd(p);
  for (int i = 0; i < p; ++i) sd(i) = std::sqrt(cov(i, i));
  Eigen::MatrixXd D(p, p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      double c = 0.0;
      const double denom = sd(i) * sd(j);
      if (denom > 0.0) c = cov(i, j) / denom;
      if (!std::isfinite(c)) c = 0.0;  // nan_to_num
      D(i, j) = 1.0 - std::abs(c);
    }
    D(i, i) = 0.0;  // guard tiny round-off on the diagonal
  }
  return D;
}

// "block_i" labels from average-linkage clustering of the correlation distance.
std::vector<std::string> correlation_block_labels(const Eigen::MatrixXd& X, int n_blocks) {
  const int p = static_cast<int>(X.cols());
  if (n_blocks < 1 || n_blocks > p)
    throw std::invalid_argument(
        "adgencov: correlation_blocks needs 1 <= n_blocks <= number of genes.");
  const Eigen::MatrixXd D = correlation_distance(X);
  const std::vector<int> codes = agglomerative_average(D, n_blocks);
  std::vector<std::string> out;
  out.reserve(codes.size());
  for (int c : codes) out.push_back("block_" + std::to_string(c));
  return out;
}

}  // namespace

std::vector<std::string> build_group_labels(const Dataset& dataset,
                                            const std::string& group,
                                            const Table* annotation,
                                            const Table* group_map,
                                            int n_blocks) {
  const auto& genes = dataset.genes;
  std::vector<std::string> out;
  out.reserve(genes.size());

  if (group == "none") {
    for (size_t i = 0; i < genes.size(); ++i) out.push_back("gene_" + std::to_string(i));
    return out;
  }
  if (group == "gene_family") {
    for (const auto& g : genes) out.push_back(gene_family_label(g));
    return out;
  }
  if (group == "reactome" || group == "go_process" || group == "custom_group_map") {
    if (!group_map)
      throw std::invalid_argument("adgencov: group '" + group +
                                  "' requires --group-map with columns gene,group.");
    auto m = table_map(*group_map, "group");
    for (const auto& g : genes) {
      auto it = m.find(g);
      out.push_back(it == m.end() ? "unmapped" : it->second);
    }
    return out;
  }
  if (group == "chromosome") {
    if (!annotation)
      throw std::invalid_argument("adgencov: chromosome grouping requires --annotation "
                                  "with columns gene,chromosome.");
    auto m = table_map(*annotation, "chromosome");
    for (const auto& g : genes) {
      auto it = m.find(g);
      out.push_back(it == m.end() ? "chr_unknown" : it->second);
    }
    return out;
  }
  if (group == "correlation_blocks") {
    return correlation_block_labels(dataset.X, n_blocks);
  }
  if (group == "hierarchical_wreath") {
    // Public approximation from the prototype: coarse mapped groups, each
    // subdivided by data-driven correlation blocks ("coarse::block_i").
    if (!group_map)
      throw std::invalid_argument(
          "adgencov: hierarchical_wreath requires --group-map with columns gene,group.");
    auto m = table_map(*group_map, "group");
    std::vector<std::string> coarse;
    coarse.reserve(genes.size());
    for (const auto& g : genes) {
      auto it = m.find(g);
      coarse.push_back(it == m.end() ? "unmapped" : it->second);
    }
    const std::vector<std::string> fine = correlation_block_labels(dataset.X, n_blocks);
    for (size_t i = 0; i < genes.size(); ++i)
      out.push_back(coarse[i] + "::" + fine[i]);
    return out;
  }
  throw std::invalid_argument("adgencov: unknown group: " + group);
}

std::vector<int> factorize(const std::vector<std::string>& labels) {
  std::unordered_map<std::string, int> code;
  std::vector<int> out;
  out.reserve(labels.size());
  for (const auto& l : labels) {
    auto it = code.find(l);
    if (it == code.end()) {
      int next = static_cast<int>(code.size());
      code.emplace(l, next);
      out.push_back(next);
    } else {
      out.push_back(it->second);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Generator-based symmetry groups.
// ---------------------------------------------------------------------------

std::vector<std::vector<int>> cyclic_generators(int p) {
  if (p < 1) throw std::invalid_argument("cyclic_generators: p must be >= 1");
  std::vector<int> shift(p);
  for (int i = 0; i < p; ++i) shift[i] = (i + 1) % p;
  return {shift};
}

std::vector<std::vector<int>> reflection_generators(int p) {
  if (p < 1) throw std::invalid_argument("reflection_generators: p must be >= 1");
  std::vector<int> rev(p);
  for (int i = 0; i < p; ++i) rev[i] = p - 1 - i;
  return {rev};
}

std::vector<std::vector<int>> dihedral_generators(int p) {
  if (p < 1) throw std::invalid_argument("dihedral_generators: p must be >= 1");
  std::vector<int> shift(p), rev(p);
  for (int i = 0; i < p; ++i) {
    shift[i] = (i + 1) % p;
    rev[i] = p - 1 - i;
  }
  return {shift, rev};
}

PairSymmetry build_symmetry(const Dataset& dataset, const std::string& group,
                            const Table* annotation, const Table* group_map,
                            int n_blocks,
                            const std::vector<std::vector<int>>* generators) {
  const int p = static_cast<int>(dataset.genes.size());

  if (group == "cyclic") {
    return pair_symmetry_from_generators(p, cyclic_generators(p));
  }
  if (group == "dihedral") {
    return pair_symmetry_from_generators(p, dihedral_generators(p));
  }
  if (group == "reflection") {
    return pair_symmetry_from_generators(p, reflection_generators(p));
  }
  if (group == "banded") {
    return pair_symmetry_banded(p);
  }
  if (group == "custom_generators") {
    if (!generators || generators->empty())
      throw std::invalid_argument(
          "adgencov: group 'custom_generators' requires --generators with one "
          "or more length-p permutations.");
    return pair_symmetry_from_generators(p, *generators);
  }

  // Everything else is a partition group: reuse the label pipeline and lift the
  // resulting block partition to a pair symmetry.
  const std::vector<int> labels =
      factorize(build_group_labels(dataset, group, annotation, group_map, n_blocks));
  return pair_symmetry_from_labels(labels);
}

}  // namespace adgencov
