# Dependencies.cmake — third-party libraries fetched at configure time.
#
# All dependencies are header-only or self-contained so that a plain
# `cmake -B build && cmake --build build` works on Linux/macOS/Windows with no
# system packages required.  Set the *_PROVIDER cache variables to "system" to
# use a locally installed copy instead of fetching.

include(FetchContent)

# ---- Eigen3 (header-only linear algebra) ---------------------------------
set(ADGENCOV_EIGEN_PROVIDER "fetch" CACHE STRING
    "Where to get Eigen3: 'fetch' (download) or 'system' (find_package)")

if(ADGENCOV_EIGEN_PROVIDER STREQUAL "system")
  find_package(Eigen3 3.4 REQUIRED NO_MODULE)
  message(STATUS "adgencov: using system Eigen3 ${Eigen3_VERSION}")
else()
  FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE)
  # Eigen's own tests/blas/etc. are heavy; disable everything but the headers.
  set(EIGEN_BUILD_DOC       OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING   OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING         OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen3)
  message(STATUS "adgencov: fetched Eigen3 3.4.0")
endif()

# ---- Catch2 (unit-test framework) ----------------------------------------
if(ADGENCOV_BUILD_TESTS)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  message(STATUS "adgencov: fetched Catch2 v3.5.4")
endif()
