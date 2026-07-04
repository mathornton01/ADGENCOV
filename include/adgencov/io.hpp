#ifndef ADGENCOV_IO_HPP
#define ADGENCOV_IO_HPP

#include <optional>
#include <string>
#include <vector>
#include <Eigen/Dense>

/// @file io.hpp
/// Tabular and expression-matrix I/O for the adgencov command-line tool.
///
/// Readers auto-detect the field delimiter (tab, comma, or run of whitespace)
/// from the header line, mirroring pandas' `sep=None` sniffing for the common
/// FPKM/TPM/count-matrix layouts the Applications Note targets.  Only plain
/// text is supported in this layer; transparent gzip is planned for a later
/// turn.

namespace adgencov {

/// A parsed delimited table: header names plus row-major string cells.
/// All rows are padded/truncated to @c headers.size() columns.
struct Table {
  std::vector<std::string> headers;
  std::vector<std::vector<std::string>> rows;

  int ncol() const { return static_cast<int>(headers.size()); }
  int nrow() const { return static_cast<int>(rows.size()); }

  /// Index of a column by (case-sensitive) name, or -1 if absent.
  int col_index(const std::string& name) const;
};

/// Read a delimited text file, sniffing the delimiter from the header line.
/// @throws std::runtime_error if the file cannot be opened or is empty.
Table read_table(const std::string& path);

/// An expression matrix selected from a raw table: one gene name per row and a
/// genes-by-samples numeric matrix over the chosen sample columns.
struct ExpressionData {
  std::vector<std::string> genes;         ///< gene name per matrix row
  std::vector<std::string> sample_cols;   ///< selected sample-column headers
  Eigen::MatrixXd values;                 ///< genes x samples (non-finite -> NaN)
};

/// Load an expression/count matrix.  Sample columns are those whose header
/// matches @c sample_regex (ECMAScript syntax, partial match).  The gene name
/// is taken from column @c gene_col if present, else the first column.  Rows
/// whose sample values are all non-finite are dropped, matching the prototype's
/// `dropna(how="all")`.
/// @throws std::runtime_error on fewer than 3 matching sample columns.
ExpressionData load_expression_matrix(const std::string& path,
                                      const std::string& sample_regex,
                                      const std::string& gene_col = "gene_short_name");

/// Write a matrix as CSV with row/column headers.  @c row_names and
/// @c col_names, when non-empty, must match @c M's dimensions.
void write_matrix_csv(const std::string& path, const Eigen::MatrixXd& M,
                      const std::vector<std::string>& row_names = {},
                      const std::vector<std::string>& col_names = {});

}  // namespace adgencov

#endif  // ADGENCOV_IO_HPP
