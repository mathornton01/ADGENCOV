#ifndef ADGENCOV_PREPROCESS_HPP
#define ADGENCOV_PREPROCESS_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>

#include "adgencov/io.hpp"

/// @file preprocess.hpp
/// Expression-matrix preprocessing for adgencov, ported from the prototype's
/// `preprocess`: duplicate-symbol collapse, low-expression filtering, variable-
/// gene selection, and per-gene z-scoring.

namespace adgencov {

/// The analysis-ready dataset: a standardized samples-by-genes matrix and the
/// gene names of its columns (in selected order).
struct Dataset {
  Eigen::MatrixXd X;                 ///< samples x genes, column-standardized
  std::vector<std::string> genes;    ///< gene name per column of X
};

/// Preprocess an expression matrix (genes x samples) into a standardized
/// samples-by-genes @c Dataset.  Steps, matching the prototype exactly:
///   1. non-finite -> 0, clip negatives to 0;
///   2. collapse duplicate gene symbols, keeping the row of largest mean;
///   3. drop genes whose mean abundance < @c min_mean;
///   4. optional log2(x + 1);
///   5. keep the @c n_genes highest-variance genes (population variance);
///   6. transpose to samples x genes, center each gene, divide by its
///      sample standard deviation (ddof=1, floored at 1e-7).
Dataset preprocess(const ExpressionData& data, int n_genes = 500,
                   double min_mean = 0.1, bool log_transform = true);

}  // namespace adgencov

#endif  // ADGENCOV_PREPROCESS_HPP
