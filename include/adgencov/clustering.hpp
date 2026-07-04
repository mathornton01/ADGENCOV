#ifndef ADGENCOV_CLUSTERING_HPP
#define ADGENCOV_CLUSTERING_HPP

#include <vector>

#include <Eigen/Dense>

/// @file clustering.hpp
/// Average-linkage agglomerative clustering on a precomputed distance matrix.
///
/// This reproduces the partition produced by
/// @c sklearn.cluster.AgglomerativeClustering(n_clusters=k,
///     metric="precomputed", linkage="average"), which the Applications Note
/// prototype uses to build data-driven "correlation_blocks" symmetry groups.

namespace adgencov {

/// UPGMA (unweighted-pair-group-method-with-arithmetic-mean) agglomerative
/// clustering on a symmetric precomputed distance matrix @p dist.
///
/// Repeatedly merges the two closest current clusters (Lance–Williams average
/// update: the distance from a merged cluster to a third is the size-weighted
/// mean of its parents' distances) until @p n_clusters clusters remain.
///
/// @param dist        square, symmetric, non-negative distance matrix (n x n).
/// @param n_clusters  number of clusters to return (1 <= n_clusters <= n).
/// @returns a length-n vector of cluster ids in [0, n_clusters); ids are
///          assigned to surviving clusters in ascending order of their
///          smallest member index (a stable canonical labeling).
/// @throws std::invalid_argument if @p dist is not square or @p n_clusters is
///          out of range.
std::vector<int> agglomerative_average(const Eigen::MatrixXd& dist, int n_clusters);

}  // namespace adgencov

#endif  // ADGENCOV_CLUSTERING_HPP
