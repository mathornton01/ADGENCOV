// adgencov — command-line driver for Algebraic-Diversity Genetic Covariance
// estimation.  Loads an expression matrix, builds a symmetry partition,
// recommends an estimator by leave-one-out NLL, and writes the covariance,
// top-covarying gene edges, and a run report.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "adgencov/groups.hpp"
#include "adgencov/io.hpp"
#include "adgencov/preprocess.hpp"
#include "adgencov/select.hpp"

namespace {

constexpr const char* kVersion = "adgencov 0.1.0";

struct Args {
  std::string expression;
  std::string sample_regex = "_LL[0-9]+";
  std::string gene_col = "gene_short_name";
  std::string group = "gene_family";
  std::string group_map;
  std::string annotation;
  int n_genes = 500;
  int n_blocks = 4;
  double min_mean = 0.1;
  double top_fraction = 0.01;
  bool log_transform = true;
  std::string outdir = "ad_covariance_output";
};

void usage(std::ostream& os) {
  os <<
      "Usage: adgencov --expression FILE [options]\n"
      "\n"
      "Required:\n"
      "  --expression FILE       Expression/count matrix (TSV/CSV/FPKM, auto-delimited)\n"
      "\n"
      "Options:\n"
      "  --sample-regex REGEX    Regex selecting sample columns (default: _LL[0-9]+)\n"
      "  --gene-col NAME         Gene-name column (default: gene_short_name, else col 0)\n"
      "  --group NAME            Symmetry partition: none | gene_family | chromosome |\n"
      "                          reactome | go_process | custom_group_map |\n"
      "                          correlation_blocks | hierarchical_wreath\n"
      "                          (default: gene_family)\n"
      "  --group-map FILE        TSV/CSV with columns gene,group (for map-based groups)\n"
      "  --annotation FILE       TSV/CSV with columns gene,chromosome\n"
      "  --n-blocks N            Blocks for correlation_blocks/wreath (default: 4)\n"
      "  --n-genes N             Keep N highest-variance genes (default: 500)\n"
      "  --min-mean X            Drop genes with mean abundance < X (default: 0.1)\n"
      "  --top-fraction X        Fraction of gene pairs to report as edges (default: 0.01)\n"
      "  --no-log                Disable log2(x+1) transform\n"
      "  --outdir DIR            Output directory (default: ad_covariance_output)\n"
      "  -h, --help              Show this help\n"
      "  --version               Show version\n";
}

// Minimal long-option parser.  Returns false on a usage/version short-circuit.
bool parse_args(int argc, char** argv, Args& a) {
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "-h" || k == "--help") { usage(std::cout); return false; }
    else if (k == "--version") { std::cout << kVersion << "\n"; return false; }
    else if (k == "--expression") a.expression = need(i);
    else if (k == "--sample-regex") a.sample_regex = need(i);
    else if (k == "--gene-col") a.gene_col = need(i);
    else if (k == "--group") a.group = need(i);
    else if (k == "--group-map") a.group_map = need(i);
    else if (k == "--annotation") a.annotation = need(i);
    else if (k == "--n-blocks") a.n_blocks = std::stoi(need(i));
    else if (k == "--n-genes") a.n_genes = std::stoi(need(i));
    else if (k == "--min-mean") a.min_mean = std::stod(need(i));
    else if (k == "--top-fraction") a.top_fraction = std::stod(need(i));
    else if (k == "--no-log") a.log_transform = false;
    else if (k == "--outdir") a.outdir = need(i);
    else throw std::runtime_error("unknown option: " + k);
  }
  if (a.expression.empty()) throw std::runtime_error("--expression is required");
  return true;
}

std::string params_json(const std::map<std::string, double>& p) {
  std::ostringstream os;
  os << "{";
  bool first = true;
  for (const auto& kv : p) {
    if (!first) os << ", ";
    first = false;
    os << "\"" << kv.first << "\": " << kv.second;
  }
  os << "}";
  return os.str();
}

// Top |covariance| off-diagonal gene pairs (prototype's top_edges).
struct Edge { std::string a, b; double cov, abscov; };

std::vector<Edge> top_edges(const Eigen::MatrixXd& S,
                            const std::vector<std::string>& genes,
                            double top_fraction) {
  std::vector<Edge> edges;
  const int p = static_cast<int>(S.rows());
  for (int i = 0; i < p; ++i)
    for (int j = i + 1; j < p; ++j)
      edges.push_back({genes[i], genes[j], S(i, j), std::abs(S(i, j))});
  std::stable_sort(edges.begin(), edges.end(),
                   [](const Edge& x, const Edge& y) { return x.abscov > y.abscov; });
  int k = std::max(1, static_cast<int>(std::lround(top_fraction * edges.size())));
  if (k < static_cast<int>(edges.size())) edges.resize(k);
  return edges;
}

// Wrap a field in quotes with RFC-4180 quote-doubling so embedded '"' survive.
std::string csv_quote(const std::string& s) {
  std::string out = "\"";
  for (char c : s) { if (c == '"') out += '"'; out += c; }
  out += '"';
  return out;
}

std::string join_path(const std::string& dir, const std::string& file) {
  if (dir.empty()) return file;
  return dir.back() == '/' ? dir + file : dir + "/" + file;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    if (!parse_args(argc, argv, args)) return 0;
  } catch (const std::exception& e) {
    std::cerr << "adgencov: " << e.what() << "\n\n";
    usage(std::cerr);
    return 2;
  }

  try {
    // --- Load + preprocess ------------------------------------------------
    adgencov::ExpressionData raw =
        adgencov::load_expression_matrix(args.expression, args.sample_regex, args.gene_col);
    adgencov::Dataset ds =
        adgencov::preprocess(raw, args.n_genes, args.min_mean, args.log_transform);
    if (ds.X.cols() < 2 || ds.X.rows() < 2)
      throw std::runtime_error("too few genes/samples survived preprocessing");

    // --- Group labels -----------------------------------------------------
    adgencov::Table ann, gmap;
    const adgencov::Table* ann_p = nullptr;
    const adgencov::Table* gmap_p = nullptr;
    if (!args.annotation.empty()) { ann = adgencov::read_table(args.annotation); ann_p = &ann; }
    if (!args.group_map.empty()) { gmap = adgencov::read_table(args.group_map); gmap_p = &gmap; }
    std::vector<std::string> label_names =
        adgencov::build_group_labels(ds, args.group, ann_p, gmap_p, args.n_blocks);
    std::vector<int> labels = adgencov::factorize(label_names);

    // --- Ensure output directory (portable, no <filesystem> dependency) ---
    std::string mkdir = "mkdir -p '" + args.outdir + "'";
    if (std::system(mkdir.c_str()) != 0)
      std::cerr << "adgencov: warning: could not create outdir " << args.outdir << "\n";

    // gene_groups.csv
    {
      std::ofstream f(join_path(args.outdir, "gene_groups.csv"));
      f << "gene,group\n";
      for (size_t i = 0; i < ds.genes.size(); ++i)
        f << ds.genes[i] << "," << label_names[i] << "\n";
    }

    // --- Recommend estimator ---------------------------------------------
    std::vector<adgencov::EstimatorResult> results =
        adgencov::recommend_estimator(ds.X, labels);
    if (results.empty()) throw std::runtime_error("no estimator could be scored");

    {
      std::ofstream f(join_path(args.outdir, "estimator_recommendations.csv"));
      f.precision(10);
      f << "rank,method,params,loo_nll,condition_number\n";
      for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        f << (i + 1) << "," << r.spec.method << "," << csv_quote(params_json(r.spec.params))
          << "," << r.loo_nll << "," << r.condition_number << "\n";
      }
    }

    const auto& best = results.front();
    adgencov::write_matrix_csv(join_path(args.outdir, "best_covariance.csv"),
                               best.covariance, ds.genes, ds.genes);

    // top_edges.csv
    std::vector<Edge> edges = top_edges(best.covariance, ds.genes, args.top_fraction);
    {
      std::ofstream f(join_path(args.outdir, "top_edges.csv"));
      f.precision(10);
      f << "gene_a,gene_b,covariance,abs_covariance\n";
      for (const auto& e : edges)
        f << e.a << "," << e.b << "," << e.cov << "," << e.abscov << "\n";
    }

    // report.md
    {
      std::ofstream f(join_path(args.outdir, "report.md"));
      f << "# ADGENCOV run report\n\n";
      f << "Samples: " << ds.X.rows() << "\n\n";
      f << "Genes: " << ds.X.cols() << "\n\n";
      f << "Selected group: " << args.group << "\n\n";
      f << "Recommended estimator: " << best.spec.method << " " << params_json(best.spec.params)
        << "\n\n";
      f << std::fixed << std::setprecision(4) << "LOOCV NLL: " << best.loo_nll << "\n\n";
      f << "See estimator_recommendations.csv, best_covariance.csv, and top_edges.csv.\n";
    }

    // --- Console summary --------------------------------------------------
    std::cout << "rank  method            loo_nll        cond\n";
    for (size_t i = 0; i < results.size() && i < 10; ++i) {
      const auto& r = results[i];
      std::cout << std::setw(4) << (i + 1) << "  " << std::left << std::setw(16)
                << r.spec.method << std::right << std::setw(12) << std::fixed
                << std::setprecision(4) << r.loo_nll << std::setw(14) << std::setprecision(2)
                << r.condition_number << "\n";
    }
    std::cout << "\nWrote outputs to " << args.outdir << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "adgencov: error: " << e.what() << "\n";
    return 1;
  }
}
