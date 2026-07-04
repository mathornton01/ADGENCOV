#include "adgencov/io.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace adgencov {

namespace {

// Sniff a single-character delimiter from a header line: prefer tab, then
// comma, then semicolon; fall back to whitespace-splitting when none appear.
char sniff_delim(const std::string& header) {
  if (header.find('\t') != std::string::npos) return '\t';
  if (header.find(',') != std::string::npos) return ',';
  if (header.find(';') != std::string::npos) return ';';
  return ' ';  // whitespace mode
}

std::vector<std::string> split_line(const std::string& line, char delim) {
  std::vector<std::string> out;
  if (delim == ' ') {
    // Collapse runs of whitespace, ignoring leading/trailing.
    std::istringstream ss(line);
    std::string tok;
    while (ss >> tok) out.push_back(tok);
    return out;
  }
  std::string cell;
  std::istringstream ss(line);
  while (std::getline(ss, cell, delim)) out.push_back(cell);
  // getline drops a trailing empty field; restore it if the line ends in delim.
  if (!line.empty() && line.back() == delim) out.emplace_back("");
  return out;
}

double parse_numeric(const std::string& s) {
  if (s.empty()) return std::numeric_limits<double>::quiet_NaN();
  try {
    size_t pos = 0;
    double v = std::stod(s, &pos);
    // Trailing non-numeric characters -> treat as missing (like to_numeric coerce).
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    if (pos != s.size()) return std::numeric_limits<double>::quiet_NaN();
    return v;
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

}  // namespace

int Table::col_index(const std::string& name) const {
  for (int i = 0; i < ncol(); ++i)
    if (headers[i] == name) return i;
  return -1;
}

Table read_table(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("adgencov: cannot open file: " + path);

  std::string line;
  // First non-empty line is the header.
  do {
    if (!std::getline(in, line)) throw std::runtime_error("adgencov: empty file: " + path);
  } while (line.empty());
  if (!line.empty() && line.back() == '\r') line.pop_back();

  Table t;
  const char delim = sniff_delim(line);
  t.headers = split_line(line, delim);
  const int ncol = static_cast<int>(t.headers.size());

  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::vector<std::string> cells = split_line(line, delim);
    cells.resize(ncol);  // pad or truncate to header width
    t.rows.push_back(std::move(cells));
  }
  return t;
}

ExpressionData load_expression_matrix(const std::string& path,
                                      const std::string& sample_regex,
                                      const std::string& gene_col) {
  Table t = read_table(path);

  int gcol = t.col_index(gene_col);
  if (gcol < 0) gcol = 0;  // fall back to first column

  std::regex re(sample_regex);
  std::vector<int> sample_idx;
  std::vector<std::string> sample_names;
  for (int c = 0; c < t.ncol(); ++c) {
    if (std::regex_search(t.headers[c], re)) {
      sample_idx.push_back(c);
      sample_names.push_back(t.headers[c]);
    }
  }
  if (sample_idx.size() < 3)
    throw std::runtime_error("adgencov: found only " + std::to_string(sample_idx.size()) +
                             " sample columns matching regex \"" + sample_regex + "\".");

  const int ns = static_cast<int>(sample_idx.size());
  std::vector<std::string> genes;
  std::vector<std::vector<double>> rows;
  genes.reserve(t.nrow());
  rows.reserve(t.nrow());

  for (const auto& r : t.rows) {
    std::vector<double> vals(ns);
    bool all_nan = true;
    for (int j = 0; j < ns; ++j) {
      double v = parse_numeric(r[sample_idx[j]]);
      vals[j] = v;
      if (std::isfinite(v)) all_nan = false;
    }
    if (all_nan) continue;  // dropna(how="all")
    genes.push_back(r[gcol]);
    rows.push_back(std::move(vals));
  }

  ExpressionData d;
  d.genes = std::move(genes);
  d.sample_cols = std::move(sample_names);
  d.values.resize(static_cast<int>(rows.size()), ns);
  for (int i = 0; i < static_cast<int>(rows.size()); ++i)
    for (int j = 0; j < ns; ++j) d.values(i, j) = rows[i][j];
  return d;
}

void write_matrix_csv(const std::string& path, const Eigen::MatrixXd& M,
                      const std::vector<std::string>& row_names,
                      const std::vector<std::string>& col_names) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("adgencov: cannot write file: " + path);
  out.precision(17);

  const bool have_rows = !row_names.empty();
  const bool have_cols = !col_names.empty();
  if (have_cols) {
    if (have_rows) out << "";  // leading empty corner cell
    for (int j = 0; j < M.cols(); ++j) {
      if (have_rows || j > 0) out << ",";
      out << col_names[j];
    }
    out << "\n";
  }
  for (int i = 0; i < M.rows(); ++i) {
    if (have_rows) out << row_names[i] << ",";
    for (int j = 0; j < M.cols(); ++j) {
      if (j > 0) out << ",";
      out << M(i, j);
    }
    out << "\n";
  }
}

}  // namespace adgencov
