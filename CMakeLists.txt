cmake_minimum_required(VERSION 3.16)

project(adgencov
  VERSION 0.1.0
  DESCRIPTION "Algebraic-Diversity Genetic Covariance estimation for transcriptomics"
  LANGUAGES CXX)

# ---- Options -------------------------------------------------------------
option(ADGENCOV_BUILD_CLI      "Build the adgencov command-line tool" ON)
option(ADGENCOV_BUILD_TESTS    "Build unit tests"                     ON)
option(ADGENCOV_BUILD_PYTHON   "Build the pybind11 Python module"     OFF)
option(ADGENCOV_USE_LAPACK     "Link system LAPACK/BLAS if available" OFF)

# ---- Standards / defaults ------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Dependencies (fetched in a later turn) ------------------------------
# Eigen3 (header-only linear algebra) — added via FetchContent/find_package.
include(cmake/Dependencies.cmake)

# ---- Core library --------------------------------------------------------
add_library(adgencov)              # sources added in src/CMakeLists.txt
target_include_directories(adgencov PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
add_library(adgencov::adgencov ALIAS adgencov)
add_subdirectory(src)

# ---- CLI -----------------------------------------------------------------
if(ADGENCOV_BUILD_CLI)
  add_subdirectory(cli)
endif()

# ---- Tests ---------------------------------------------------------------
if(ADGENCOV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Python bindings -----------------------------------------------------
if(ADGENCOV_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()

message(STATUS "adgencov ${PROJECT_VERSION} configured "
  "(CLI=${ADGENCOV_BUILD_CLI} TESTS=${ADGENCOV_BUILD_TESTS} PYTHON=${ADGENCOV_BUILD_PYTHON})")
