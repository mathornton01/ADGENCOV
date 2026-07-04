#ifndef ADGENCOV_GROUPS_HPP
#define ADGENCOV_GROUPS_HPP

#include <string>
#include <vector>

#include "adgencov/io.hpp"
#include "adgencov/preprocess.hpp"

/// @file groups.hpp
/// Biologically meaningful symmetry-group builders for adgencov.
///
/// These are the public, user-selectable partitions from the Applications Note
/// prototype.  Each returns one group label per gene (in the column order of
/// @c dataset).  Data-driven partitions that require clustering
/// (correlation_blocks, hierarchical_wreath) are added in a later turn.

namespace adgencov {

/// Transparent gene-family heuristic: a curated prefix list, else the leading
/// alphabetic run (truncated to 4 chars).  Ported verbatim from the prototype.
std::string gene_family_label(const std::string& gene);

/// Build per-gene group labels for the named partition.
///   * "none"             -> one singleton group per gene ("gene_i");
///   * "gene_family"      -> @c gene_family_label per gene;
///   * "chromosome"       -> requires @c annotation (columns gene,chromosome);
///   * "reactome" / "go_process" / "custom_group_map"
///                        -> requires @c group_map (columns gene,group).
/// Unmapped genes get "unmapped" (group map) or "chr_unknown" (chromosome).
/// @param annotation  optional table, or nullptr.
/// @param group_map   optional table, or nullptr.
/// @throws std::invalid_argument for an unknown or not-yet-supported group, or
///         a required table that is missing / lacks the expected columns.
std::vector<std::string> build_group_labels(const Dataset& dataset,
                                            const std::string& group,
                                            const Table* annotation = nullptr,
                                            const Table* group_map = nullptr);

/// Factorize string labels into dense integer codes (0..k-1) assigned in order
/// of first appearance — the integer labels consumed by the numerics layer.
std::vector<int> factorize(const std::vector<std::string>& labels);

}  // namespace adgencov

#endif  // ADGENCOV_GROUPS_HPP
