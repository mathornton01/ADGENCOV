// Tests for the adgencov I/O layer: delimiter sniffing, sample-column
// selection by regex, non-finite handling, and CSV matrix round-trips.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdio>
#include <fstream>
#include <string>

#include "adgencov/io.hpp"

#ifndef ADGENCOV_TEST_DIR
#define ADGENCOV_TEST_DIR "."
#endif

using Catch::Matchers::WithinAbs;
using namespace adgencov;

namespace {
std::string write_temp(const std::string& name, const std::string& content) {
  std::string path = std::string(ADGENCOV_TEST_DIR) + "/_tmp_" + name;
  std::ofstream f(path);
  f << content;
  return path;
}
}  // namespace

TEST_CASE("read_table sniffs tab, comma, and whitespace delimiters", "[io]") {
  {
    Table t = read_table(write_temp("tab.tsv", "gene\ta\tb\nG1\t1\t2\nG2\t3\t4\n"));
    REQUIRE(t.ncol() == 3);
    REQUIRE(t.nrow() == 2);
    REQUIRE(t.headers[0] == "gene");
    REQUIRE(t.col_index("b") == 2);
    REQUIRE(t.col_index("missing") == -1);
    REQUIRE(t.rows[1][2] == "4");
  }
  {
    Table t = read_table(write_temp("csv.csv", "gene,a,b\nG1,1,2\n"));
    REQUIRE(t.ncol() == 3);
    REQUIRE(t.rows[0][1] == "1");
  }
  {
    Table t = read_table(write_temp("ws.txt", "gene a b\nG1 1 2\n"));
    REQUIRE(t.ncol() == 3);
    REQUIRE(t.rows[0][2] == "2");
  }
}

TEST_CASE("load_expression_matrix selects sample columns by regex", "[io]") {
  const std::string content =
      "gene_short_name\tchrom\tDex_LL01\tDex_LL02\tUntreated_LL03\tmeta\n"
      "G1\tchr1\t1.0\t2.0\t3.0\tx\n"
      "G2\tchr2\t4.0\t5.0\t6.0\ty\n";
  std::string path = write_temp("expr.tsv", content);

  ExpressionData d = load_expression_matrix(path, "_LL[0-9]+", "gene_short_name");
  REQUIRE(d.sample_cols.size() == 3);
  REQUIRE(d.sample_cols[0] == "Dex_LL01");
  REQUIRE(d.genes.size() == 2);
  REQUIRE(d.genes[0] == "G1");
  REQUIRE(d.values.rows() == 2);
  REQUIRE(d.values.cols() == 3);
  REQUIRE_THAT(d.values(1, 2), WithinAbs(6.0, 1e-12));
}

TEST_CASE("load_expression_matrix falls back to first column and drops all-NaN rows",
          "[io]") {
  const std::string content =
      "id\tS_LL1\tS_LL2\tS_LL3\n"
      "G1\t1\t2\t3\n"
      "G2\tNA\t\tnan\n";  // all non-numeric -> dropped
  std::string path = write_temp("expr2.tsv", content);
  ExpressionData d = load_expression_matrix(path, "_LL[0-9]+");
  REQUIRE(d.genes.size() == 1);
  REQUIRE(d.genes[0] == "G1");
}

TEST_CASE("load_expression_matrix throws on too few sample columns", "[io]") {
  std::string path = write_temp("expr3.tsv", "gene\tX_LL1\tother\nG1\t1\t2\n");
  REQUIRE_THROWS(load_expression_matrix(path, "_LL[0-9]+"));
}

TEST_CASE("write_matrix_csv round-trips values", "[io]") {
  Eigen::MatrixXd M(2, 2);
  M << 1.5, -2.25, 3.0, 4.125;
  std::string path = std::string(ADGENCOV_TEST_DIR) + "/_tmp_mat.csv";
  write_matrix_csv(path, M, {"r0", "r1"}, {"c0", "c1"});

  Table t = read_table(path);
  REQUIRE(t.headers.size() == 3);          // empty corner + 2 col names
  REQUIRE(t.headers[1] == "c0");
  REQUIRE(t.rows[0][0] == "r0");
  REQUIRE_THAT(std::stod(t.rows[1][2]), WithinAbs(4.125, 1e-12));
}
