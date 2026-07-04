# Core library sources for adgencov::adgencov (target declared in the root
# CMakeLists.txt).
target_sources(adgencov PRIVATE
  adgencov/projection.cpp
  adgencov/shrink.cpp
  adgencov/select.cpp
  adgencov/io.cpp
  adgencov/preprocess.cpp
  adgencov/groups.cpp
  adgencov/clustering.cpp)

target_link_libraries(adgencov PUBLIC Eigen3::Eigen)

# ---- Host-native vectorization ------------------------------------------
# Compile the numerical core with -march=native so Eigen emits AVX2/FMA in the
# Cholesky / triangular-solve / matmul hot paths that dominate loo_nll.  PUBLIC
# so the pybind11 module inherits it.  Opt out with -DADGENCOV_NATIVE_ARCH=OFF
# for a portable/redistributable binary.
option(ADGENCOV_NATIVE_ARCH "Compile the core with -march=native" ON)
if(ADGENCOV_NATIVE_ARCH AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" ADGENCOV_HAS_MARCH_NATIVE)
  if(ADGENCOV_HAS_MARCH_NATIVE)
    target_compile_options(adgencov PUBLIC -march=native)
    message(STATUS "adgencov: -march=native enabled for the numerical core")
  endif()
endif()
