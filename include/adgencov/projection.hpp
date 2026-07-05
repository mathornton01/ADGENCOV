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

/// A partition of the @c p x @c p index pairs into orbits (equivalence classes),
/// defining a general Reynolds/commutant projection:
///
///   P(i,j) = mean of S over every pair (k,l) in the orbit of (i,j).
///
/// This is the single primitive that generalises @c reynolds_project beyond
/// block exchangeability.  Any finite permutation group @c G acting on the @c p
/// variables induces such a partition (the orbits of @c G on ordered index
/// pairs), and averaging @c S over those orbits is exactly the orthogonal
/// (Frobenius) projection onto the commutant of @c G — the algebra of matrices
/// invariant under @c G.  Directly-specified structural classes (e.g. banded /
/// Toeplitz, where the class of (i,j) is |i-j|) are handled by the same engine.
///
/// The block-exchangeable projection above is the special case @c G = a Young
/// subgroup (permute freely within each block); @c pair_symmetry_from_labels
/// reproduces it bit-for-bit up to summation order.
struct PairSymmetry {
  int p = 0;                    ///< number of variables
  int n_orbits = 0;             ///< number of distinct orbits
  std::vector<int> orbit_of;    ///< length p*p, row-major; orbit id of (i,j)
};

/// Block-exchangeable symmetry from per-gene block @c labels (Young subgroup).
/// Produces the same three orbit kinds the block projection averages over:
/// within-block diagonals, within-block off-diagonals, and each ordered
/// cross-block rectangle.
/// @throws std::invalid_argument if @c labels is empty.
PairSymmetry pair_symmetry_from_labels(const std::vector<int>& labels);

/// Symmetry from permutation @c generators.  Each generator is a length-@c p
/// permutation given as images (generator[i] is where variable @c i maps).  The
/// orbits of index pairs under the generated group are computed by union-find
/// flood-fill over the generators — the full group is never enumerated, so the
/// cost is O(p^2 * #generators * alpha).
/// @throws std::invalid_argument if @c p < 1 or any generator is not a valid
///         permutation of [0, p).
PairSymmetry pair_symmetry_from_generators(
    int p, const std::vector<std::vector<int>>& generators);

/// Banded / symmetric-Toeplitz structure: pairs sharing the same band |i-j|
/// form one orbit (p orbits total).  This is the projection onto symmetric
/// Toeplitz matrices — covariance depending on positional distance, not
/// identity.  (It coincides with the cyclic-group commutant only in the
/// wrap-around / circulant case; here it is the plain, non-circulant banding.)
/// @throws std::invalid_argument if @c p < 1.
PairSymmetry pair_symmetry_banded(int p);

/// General orbit-averaging Reynolds projection: set every entry of @c P to the
/// mean of @c S over its orbit in @c sym.  The result is symmetric (exactly, up
/// to a final symmetrisation) and is the orthogonal projection of @c S onto the
/// subspace of matrices constant on @c sym's orbits.
/// @throws std::invalid_argument if @c S is not square or @c sym.p != S.rows().
Eigen::MatrixXd reynolds_project(const Eigen::MatrixXd& S,
                                 const PairSymmetry& sym);

}  // namespace adgencov

#endif  // ADGENCOV_PROJECTION_HPP
