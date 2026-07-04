#ifndef ADGENCOV_PROJECTION_HPP
#define ADGENCOV_PROJECTION_HPP

#include <vector>
#include <Eigen/Dense>

namespace adgencov {

/// Project a covariance matrix onto the block-exchangeable commutant.
///
/// This is the Reynolds (group-averaging) projection for a symmetry group that
/// permutes genes independently within each block of a partition.  The
/// projected matrix @c P is the orthogonal projection of @c S (in the Frobenius
/// inner product) onto the algebra of matrices that commute with every
/// within-block permutation.  Concretely, for a partition of the @c p variables
/// into blocks it averages:
///   (i)   the diagonal entries within each block,
///   (ii)  the off-diagonal entries within each block, and
///   (iii) all cross-block entries for each ordered pair of blocks.
///
/// The result is symmetric and shares the eigen-structure implied by the
/// symmetry, which reduces estimator variance when the assumed exchangeability
/// approximately holds.
///
/// @param S       A @c p x @c p symmetric covariance matrix.
/// @param labels  Length-@c p vector of integer block ids (any integers; only
///                equality matters).  @c labels[i] is the block of variable @c i.
/// @returns       The @c p x @c p projected, symmetric covariance matrix.
/// @throws std::invalid_argument if sizes disagree or @c S is not square.
Eigen::MatrixXd reynolds_project(const Eigen::MatrixXd& S,
                                 const std::vector<int>& labels);

}  // namespace adgencov

#endif  // ADGENCOV_PROJECTION_HPP
