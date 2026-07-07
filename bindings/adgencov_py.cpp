// adgencov_py.cpp — pybind11 bindings for the ADGENCOV C++ core.
//
// Exposes the full numerical + I/O pipeline (projection, estimator family,
// model selection, clustering, expression I/O, preprocessing, group builders)
// to Python.  Eigen matrices/vectors convert transparently to and from NumPy
// arrays via pybind11/eigen.h, so callers work in idiomatic NumPy while the
// heavy lifting runs in vectorised C++.
//
// This module is the language boundary for the higher layers of the product:
// the GEO-ingestion module, the FastAPI service, and the web/desktop GUIs all
// drive the fast path through `import adgencov` (which re-exports `_core`).
//
// The importable name is `adgencov._core`.  Every function mirrors the
// prototype ad_covariance_app.py exactly and is parity-tested to ~1e-9 in
// tests/test_bindings.py.

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "adgencov/projection.hpp"
#include "adgencov/shrink.hpp"
#include "adgencov/select.hpp"
#include "adgencov/clustering.hpp"
#include "adgencov/io.hpp"
#include "adgencov/preprocess.hpp"
#include "adgencov/groups.hpp"

namespace py = pybind11;
using namespace adgencov;

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "ADGENCOV C++ core (Algebraic-Diversity Genetic Covariance).\n"
      "Vectorised Eigen implementation of the Applications Note pipeline, "
      "bound to NumPy via pybind11.  Import `adgencov` for the friendly API.";
  m.attr("__version__") = "0.1.0";

  // ---- projection.hpp ----------------------------------------------------
  m.def("reynolds_project",
        py::overload_cast<const Eigen::MatrixXd&, const std::vector<int>&>(
            &reynolds_project),
        py::arg("S"), py::arg("labels"),
        R"doc(Project a covariance onto the block-exchangeable commutant.

The Reynolds (group-averaging) projection for independent within-block
permutation symmetries: averages within-block diagonals, within-block
off-diagonals, and each ordered pair of cross-block entries.

Parameters
----------
S : (p, p) float64 ndarray
    Symmetric covariance matrix.
labels : sequence of int, length p
    Block id per variable (only equality matters).

Returns
-------
(p, p) ndarray
    The symmetric projected covariance.)doc");

  // General orbit-averaging (commutant) projection + the symmetry it needs.
  py::class_<PairSymmetry>(m, "PairSymmetry",
                           "A partition of the p*p index pairs into orbits, "
                           "defining a general Reynolds/commutant projection "
                           "(P(i,j) = mean of S over the orbit of (i,j)).")
      .def(py::init<>())
      .def_readwrite("p", &PairSymmetry::p)
      .def_readwrite("n_orbits", &PairSymmetry::n_orbits)
      .def_readwrite("orbit_of", &PairSymmetry::orbit_of)
      .def("__repr__", [](const PairSymmetry& s) {
        return "PairSymmetry(p=" + std::to_string(s.p) +
               ", n_orbits=" + std::to_string(s.n_orbits) + ")";
      });

  m.def("reynolds_project",
        py::overload_cast<const Eigen::MatrixXd&, const PairSymmetry&>(
            &reynolds_project),
        py::arg("S"), py::arg("sym"),
        "Project S onto the commutant defined by a PairSymmetry: average S over "
        "each orbit of index pairs.  Generalises the labels form to any finite "
        "permutation symmetry (cyclic, dihedral, reflection, banded, custom).");
  m.def("pair_symmetry_from_labels", &pair_symmetry_from_labels,
        py::arg("labels"),
        "Block-exchangeable (Young subgroup) symmetry from per-gene labels; "
        "reproduces the labels-based reynolds_project up to summation order.");
  m.def("pair_symmetry_from_generators", &pair_symmetry_from_generators,
        py::arg("p"), py::arg("generators"),
        "Symmetry from permutation generators (each a length-p permutation of "
        "[0,p)); pair orbits are computed by union-find over the generators.");
  m.def("pair_symmetry_banded", &pair_symmetry_banded, py::arg("p"),
        "Banded / symmetric-Toeplitz symmetry: pairs sharing |i-j| form one "
        "orbit (positional-distance structure).");

  // ---- shrink.hpp --------------------------------------------------------
  m.def("sample_covariance", &sample_covariance, py::arg("X"),
        py::arg("unbiased") = true,
        "Sample covariance of a samples-by-genes matrix X (n rows, p cols). "
        "unbiased=True divides by (n-1) (numpy.cov default); False divides by n.");
  m.def("make_pd", &make_pd, py::arg("S"), py::arg("floor") = 1e-5,
        "Nearest SPD matrix by flooring eigenvalues at `floor` (symmetrised first).");
  m.def("ridge", &ridge, py::arg("S"), py::arg("alpha"),
        "Diagonal-loading shrinkage: (1-alpha)*S + alpha*(tr(S)/p)*I.");
  m.def("soft_threshold_offdiag", &soft_threshold_offdiag, py::arg("S"),
        py::arg("lam"), py::arg("l1_ratio") = 1.0,
        "Soft-threshold off-diagonals (LASSO/elastic-net covariance).");
  m.def("shrink_to_target", &shrink_to_target, py::arg("S"), py::arg("target"),
        py::arg("lambda"),
        "Convex shrinkage toward an explicit target: (1-lambda)*S + lambda*target "
        "(lambda clamped to [0,1]).  The symmetry-target ('AD-target') op with "
        "target = the Reynolds projection P_G(S).");
  m.def("ledoit_wolf_shrinkage", &ledoit_wolf_shrinkage, py::arg("X"),
        "Ledoit-Wolf optimal shrinkage intensity in [0,1] (matches scikit-learn).");
  m.def("ledoit_wolf", &ledoit_wolf, py::arg("X"),
        "Ledoit-Wolf covariance estimate from data X.");
  m.def("oas_shrinkage", &oas_shrinkage, py::arg("X"),
        "Oracle Approximating Shrinkage intensity in [0,1] (matches scikit-learn).");
  m.def("oas", &oas, py::arg("X"), "OAS covariance estimate from data X.");

  // ---- select.hpp --------------------------------------------------------
  py::class_<EstimatorSpec>(m, "EstimatorSpec",
                            "A candidate estimator: a method name plus scalar "
                            "hyper-parameters (e.g. alpha, lam, l1_ratio).")
      .def(py::init<>())
      .def(py::init([](std::string method, std::map<std::string, double> params) {
             return EstimatorSpec{std::move(method), std::move(params)};
           }),
           py::arg("method"),
           py::arg("params") = std::map<std::string, double>{})
      .def_readwrite("method", &EstimatorSpec::method)
      .def_readwrite("params", &EstimatorSpec::params)
      .def("__repr__", [](const EstimatorSpec& s) {
        std::string r = "EstimatorSpec(method='" + s.method + "', params={";
        bool first = true;
        for (const auto& kv : s.params) {
          if (!first) r += ", ";
          r += "'" + kv.first + "': " + std::to_string(kv.second);
          first = false;
        }
        return r + "})";
      });

  py::class_<EstimatorResult>(m, "EstimatorResult",
                              "One scored candidate: spec, full-data SPD "
                              "covariance, LOO-NLL, and 2-norm condition number.")
      .def_readonly("spec", &EstimatorResult::spec)
      .def_readonly("covariance", &EstimatorResult::covariance)
      .def_readonly("loo_nll", &EstimatorResult::loo_nll)
      .def_readonly("condition_number", &EstimatorResult::condition_number)
      .def("__repr__", [](const EstimatorResult& r) {
        return "EstimatorResult(method='" + r.spec.method +
               "', loo_nll=" + std::to_string(r.loo_nll) +
               ", cond=" + std::to_string(r.condition_number) + ")";
      });

  py::enum_<SelectionCriterion>(m, "SelectionCriterion",
                                "Model-selection criterion for recommend_estimator.")
      .value("Loo", SelectionCriterion::Loo, "Exact leave-one-out CV NLL (default).")
      .value("Ebic", SelectionCriterion::Ebic, "Extended BIC penalized likelihood.")
      .value("Kfold", SelectionCriterion::Kfold, "k-fold CV NLL.");

  py::class_<CriterionSpec>(m, "CriterionSpec",
                            "Selection criterion plus its hyper-parameters; "
                            "defaults to leave-one-out.")
      .def(py::init<>())
      .def(py::init([](SelectionCriterion type, double gamma, int k) {
             return CriterionSpec{type, gamma, k};
           }),
           py::arg("type") = SelectionCriterion::Loo, py::arg("gamma") = 0.5,
           py::arg("k") = 5)
      .def_readwrite("type", &CriterionSpec::type)
      .def_readwrite("gamma", &CriterionSpec::gamma)
      .def_readwrite("k", &CriterionSpec::k);

  // Bound with the struct spec...
  m.def("estimate_covariance",
        py::overload_cast<const Eigen::MatrixXd&, const std::vector<int>&,
                          const EstimatorSpec&>(&estimate_covariance),
        py::arg("X"), py::arg("labels"), py::arg("spec"),
        py::call_guard<py::gil_scoped_release>(),
        "Dispatch to the named estimator; AD variants project first; result "
        "is always make_pd'd.");
  // ...and a convenience overload taking (method, params) directly.
  m.def(
      "estimate_covariance",
      [](const Eigen::MatrixXd& X, const std::vector<int>& labels,
         const std::string& method, std::map<std::string, double> params) {
        return estimate_covariance(X, labels, EstimatorSpec{method, std::move(params)});
      },
      py::arg("X"), py::arg("labels"), py::arg("method"),
      py::arg("params") = std::map<std::string, double>{},
      py::call_guard<py::gil_scoped_release>(),
      "Convenience form: estimate_covariance(X, labels, method, params_dict).");

  m.def("gaussian_nll_one", &gaussian_nll_one, py::arg("x"), py::arg("mu"),
        py::arg("Sigma"),
        "Per-observation multivariate-Gaussian NLL (Sigma passed through make_pd).");

  m.def("loo_nll",
        py::overload_cast<const Eigen::MatrixXd&, const std::vector<int>&,
                          const EstimatorSpec&>(&loo_nll),
        py::arg("X"), py::arg("labels"), py::arg("spec"),
        py::call_guard<py::gil_scoped_release>(),
        "Leave-one-out CV mean NLL (rank-1 downdate fast path for covariance-only "
        "methods).");
  m.def(
      "loo_nll",
      [](const Eigen::MatrixXd& X, const std::vector<int>& labels,
         const std::string& method, std::map<std::string, double> params) {
        return loo_nll(X, labels, EstimatorSpec{method, std::move(params)});
      },
      py::arg("X"), py::arg("labels"), py::arg("method"),
      py::arg("params") = std::map<std::string, double>{},
      py::call_guard<py::gil_scoped_release>(),
      "Convenience form: loo_nll(X, labels, method, params_dict).");

  // Penalized-likelihood (Extended BIC) selection criterion — a single
  // full-sample pass per candidate, no refit loop.
  m.def("ebic_score",
        py::overload_cast<const Eigen::MatrixXd&, const std::vector<int>&,
                          const EstimatorSpec&, double>(&ebic_score),
        py::arg("X"), py::arg("labels"), py::arg("spec"), py::arg("gamma") = 0.5,
        py::call_guard<py::gil_scoped_release>(),
        "Extended BIC (Foygel & Drton) of a candidate: -2*loglik + dim*log(n) + "
        "4*gamma*edges*log(p), computed in one full-sample pass. Lower is better.");
  m.def(
      "ebic_score",
      [](const Eigen::MatrixXd& X, const std::vector<int>& labels,
         const std::string& method, std::map<std::string, double> params,
         double gamma) {
        return ebic_score(X, labels, EstimatorSpec{method, std::move(params)}, gamma);
      },
      py::arg("X"), py::arg("labels"), py::arg("method"),
      py::arg("params") = std::map<std::string, double>{}, py::arg("gamma") = 0.5,
      py::call_guard<py::gil_scoped_release>(),
      "Convenience form: ebic_score(X, labels, method, params_dict, gamma).");

  m.def(
      "effective_df",
      [](const EstimatorSpec& spec, const std::vector<int>& labels,
         const Eigen::MatrixXd& Sigma, double tol) {
        const PairSymmetry sym = pair_symmetry_from_labels(labels);
        int edges = 0;
        const int dim = effective_df(spec, sym, Sigma, tol, &edges);
        return std::make_pair(dim, edges);
      },
      py::arg("spec"), py::arg("labels"), py::arg("Sigma"), py::arg("tol") = 1e-8,
      "Effective model dimension (total, edges) of an estimate under the "
      "block symmetry from `labels`; the df the EBIC penalty charges for.");

  // k-fold cross-validated NLL selection criterion (contiguous unshuffled folds).
  m.def("kfold_nll",
        py::overload_cast<const Eigen::MatrixXd&, const std::vector<int>&,
                          const EstimatorSpec&, int>(&kfold_nll),
        py::arg("X"), py::arg("labels"), py::arg("spec"), py::arg("k"),
        py::call_guard<py::gil_scoped_release>(),
        "k-fold CV mean Gaussian NLL (k contiguous folds; k refits per candidate).");
  m.def(
      "kfold_nll",
      [](const Eigen::MatrixXd& X, const std::vector<int>& labels,
         const std::string& method, std::map<std::string, double> params, int k) {
        return kfold_nll(X, labels, EstimatorSpec{method, std::move(params)}, k);
      },
      py::arg("X"), py::arg("labels"), py::arg("method"),
      py::arg("params") = std::map<std::string, double>{}, py::arg("k"),
      py::call_guard<py::gil_scoped_release>(),
      "Convenience form: kfold_nll(X, labels, method, params_dict, k).");

  m.def("candidate_grid", &candidate_grid, py::arg("p"), py::arg("n"),
        "The conservative candidate grid for a (p genes, n samples) problem.");
  m.def("recommend_estimator",
        py::overload_cast<const Eigen::MatrixXd&, const std::vector<int>&>(
            &recommend_estimator),
        py::arg("X"), py::arg("labels"),
        py::call_guard<py::gil_scoped_release>(),
        "Score the candidate grid and return results sorted by ascending LOO-NLL.");
  m.def("recommend_estimator",
        py::overload_cast<const Eigen::MatrixXd&, const PairSymmetry&>(
            &recommend_estimator),
        py::arg("X"), py::arg("sym"),
        py::call_guard<py::gil_scoped_release>(),
        "General-symmetry overload: score the grid projecting AD variants through "
        "the commutant of an arbitrary PairSymmetry, sorted ascending by LOO-NLL.");
  m.def("recommend_estimator",
        py::overload_cast<const Eigen::MatrixXd&, const std::vector<int>&,
                          const CriterionSpec&>(&recommend_estimator),
        py::arg("X"), py::arg("labels"), py::arg("criterion"),
        py::call_guard<py::gil_scoped_release>(),
        "Rank the grid by the given CriterionSpec (loo/ebic/kfold); the "
        "EstimatorResult.loo_nll field carries the chosen criterion's score.");
  m.def("recommend_estimator",
        py::overload_cast<const Eigen::MatrixXd&, const PairSymmetry&,
                          const CriterionSpec&>(&recommend_estimator),
        py::arg("X"), py::arg("sym"), py::arg("criterion"),
        py::call_guard<py::gil_scoped_release>(),
        "General-symmetry, criterion-parameterized overload of recommend_estimator.");

  // ---- clustering.hpp ----------------------------------------------------
  m.def("agglomerative_average", &agglomerative_average, py::arg("dist"),
        py::arg("n_clusters"),
        "UPGMA average-linkage clustering on a precomputed distance matrix "
        "(matches sklearn AgglomerativeClustering metric='precomputed').");

  // ---- io.hpp ------------------------------------------------------------
  py::class_<Table>(m, "Table", "A parsed delimited table (headers + string cells).")
      .def_readonly("headers", &Table::headers)
      .def_readonly("rows", &Table::rows)
      .def_property_readonly("ncol", &Table::ncol)
      .def_property_readonly("nrow", &Table::nrow)
      .def("col_index", &Table::col_index, py::arg("name"));

  py::class_<ExpressionData>(m, "ExpressionData",
                             "A genes-by-samples expression matrix with names.")
      .def_readonly("genes", &ExpressionData::genes)
      .def_readonly("sample_cols", &ExpressionData::sample_cols)
      .def_readonly("values", &ExpressionData::values);

  m.def("read_table", &read_table, py::arg("path"),
        "Read a delimited text file, sniffing the delimiter from the header.");
  m.def("load_expression_matrix", &load_expression_matrix, py::arg("path"),
        py::arg("sample_regex"), py::arg("gene_col") = "gene_short_name",
        py::call_guard<py::gil_scoped_release>(),
        "Load an expression/count matrix; sample columns match sample_regex.");
  m.def("write_matrix_csv", &write_matrix_csv, py::arg("path"), py::arg("M"),
        py::arg("row_names") = std::vector<std::string>{},
        py::arg("col_names") = std::vector<std::string>{},
        "Write a matrix as CSV with optional row/column headers.");

  // ---- preprocess.hpp ----------------------------------------------------
  py::class_<Dataset>(m, "Dataset",
                      "Analysis-ready standardized samples-by-genes matrix + names.")
      .def_readonly("X", &Dataset::X)
      .def_readonly("genes", &Dataset::genes);

  m.def("preprocess", &preprocess, py::arg("data"), py::arg("n_genes") = 500,
        py::arg("min_mean") = 0.1, py::arg("log_transform") = true,
        py::call_guard<py::gil_scoped_release>(),
        "Preprocess a genes-by-samples ExpressionData into a standardized "
        "samples-by-genes Dataset.");

  // ---- groups.hpp --------------------------------------------------------
  m.def("gene_family_label", &gene_family_label, py::arg("gene"),
        "Transparent gene-family heuristic label for a gene symbol.");
  m.def("build_group_labels", &build_group_labels, py::arg("dataset"),
        py::arg("group"), py::arg("annotation") = nullptr,
        py::arg("group_map") = nullptr, py::arg("n_blocks") = 4,
        py::return_value_policy::move,
        "Build per-gene string group labels for the named partition. "
        "annotation/group_map are Table objects (or None).");
  m.def("factorize", &factorize, py::arg("labels"),
        "Factorize string labels into dense integer codes (first-appearance order).");
}
