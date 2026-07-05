#ifndef ADGENCOV_GROUPS_HPP
#define ADGENCOV_GROUPS_HPP

#include <string>
#include <vector>

#include "adgencov/io.hpp"
#include "adgencov/preprocess.hpp"
#include "adgencov/projection.hpp"  // PairSymmetry

/// @file groups.hpp
/// Biologically meaningful symmetry-group builders for adgencov.
///
/// These are the public, user-selectable partitions from the Applications Note
/// prototype.  Each returns one group label per gene (in the column order of
/// @c dataset), including the data-driven partitions that cluster the
/// gene-gene correlation structure (correlation_blocks, hierarchical_wreath).

namespace adgencov {

/// Transparent gene-family heuristic: a curated prefix list, else the leading
/// alphabetic run (truncated to 4 chars).  Ported verbatim from the prototype.
std::string gene_family_label(const std::string& gene);

/// Build per-gene group labels for the named partition.
///   * "none"             -> one singleton group per gene ("gene_i");
///   * "gene_family"      -> @c gene_family_label per gene;
///   * "chromosome"       -> requires @c annotation (columns gene,chromosome);
///   * "reactome" / "go_process" / "custom_group_map"
///                        -> requires @c group_map (columns gene,group);
///   * "correlation_blocks" -> data-driven: average-linkage clustering of the
///                        gene-gene correlation distance (1 - |corr|) into
///                        @c n_blocks blocks ("block_i");
///   * "hierarchical_wreath" -> requires @c group_map; nests correlation
///                        blocks within each mapped group ("coarse::block_i").
/// Unmapped genes get "unmapped" (group map) or "chr_unknown" (chromosome).
/// @param annotation  optional table, or nullptr.
/// @param group_map   optional table, or nullptr.
/// @param n_blocks    number of correlation blocks for data-driven groups.
/// @throws std::invalid_argument for an unknown group, a required table that is
///         missing / lacks the expected columns, or n_blocks out of range.
std::vector<std::string> build_group_labels(const Dataset& dataset,
                                            const std::string& group,
                                            const Table* annotation = nullptr,
                                            const Table* group_map = nullptr,
                                            int n_blocks = 4);

/// Factorize string labels into dense integer codes (0..k-1) assigned in order
/// of first appearance — the integer labels consumed by the numerics layer.
std::vector<int> factorize(const std::vector<std::string>& labels);

// ---------------------------------------------------------------------------
// Generator-based symmetry groups.
//
// These are the symmetries the partition/label pipeline cannot express: they
// act on the *ordering* of the genes rather than a partition of them.  Each
// returns a set of permutation generators (length-p images) for
// pair_symmetry_from_generators.  The genes are taken in their column order, so
// "position" means the gene's index in the analysed matrix.
// ---------------------------------------------------------------------------

/// Cyclic group C_p: one generator, the shift i -> (i+1) mod p.  Its commutant
/// is the circulant matrices — ring / wrap-around positional structure.
std::vector<std::vector<int>> cyclic_generators(int p);

/// Dihedral group D_p: the cyclic shift plus the reflection i -> (p-1-i).
/// Ring structure that is also invariant under reversal / reflection.
std::vector<std::vector<int>> dihedral_generators(int p);

/// Reflection group Z_2: one generator, the reversal i -> (p-1-i).  Palindromic
/// / mirror-symmetric positional structure.
std::vector<std::vector<int>> reflection_generators(int p);

/// Build the general pair symmetry for the named @c group over the dataset's
/// genes (column order).  Extends @c build_group_labels with the non-partition
/// groups:
///   * "cyclic" / "dihedral" / "reflection" -> generator groups above;
///   * "banded" -> symmetric-Toeplitz (positional-distance) structure;
///   * "custom_generators" -> requires @c generators (each a length-p
///                            permutation of [0,p));
/// and reproduces every partition group ("none", "gene_family", "chromosome",
/// "reactome"/…, "correlation_blocks", "hierarchical_wreath") via
/// pair_symmetry_from_labels(factorize(build_group_labels(...))).
/// @throws std::invalid_argument for an unknown group, a missing required table
///         / generators, or generators of the wrong length.
PairSymmetry build_symmetry(const Dataset& dataset, const std::string& group,
                            const Table* annotation = nullptr,
                            const Table* group_map = nullptr, int n_blocks = 4,
                            const std::vector<std::vector<int>>* generators = nullptr);

}  // namespace adgencov

#endif  // ADGENCOV_GROUPS_HPP
