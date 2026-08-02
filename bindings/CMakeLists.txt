# bindings/CMakeLists.txt — the pybind11 Python extension module.
#
# Builds `adgencov._core`, a thin wrapper over the C++ core library.  The
# compiled module is placed directly into python/adgencov/ so the pure-Python
# package (python/adgencov/__init__.py) can `from . import _core` after an
# editable install or a simple `PYTHONPATH=python` for development.

pybind11_add_module(adgencov_core MODULE adgencov_py.cpp)

# The importable name is `_core`; the CMake target is `adgencov_core` to avoid
# colliding with the `adgencov` core library target.
set_target_properties(adgencov_core PROPERTIES OUTPUT_NAME _core)

target_link_libraries(adgencov_core PRIVATE adgencov::adgencov)
target_compile_features(adgencov_core PRIVATE cxx_std_17)

# Drop the built module next to the Python sources so it is importable in place.
set(ADGENCOV_PY_PKG_DIR ${CMAKE_SOURCE_DIR}/python/adgencov)
set_target_properties(adgencov_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${ADGENCOV_PY_PKG_DIR})

# Multi-config generators (Visual Studio) append the configuration name to the
# output directory, landing the module at python/adgencov/Release/_core.pyd
# where `from . import _core` cannot find it — Python then reports the confusing
# "partially initialized module ... circular import".  Pin every configuration
# to the package directory so the module is importable in place on Windows too.
foreach(_cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${_cfg}" _CFG)
  set_target_properties(adgencov_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${_CFG} ${ADGENCOV_PY_PKG_DIR}
    RUNTIME_OUTPUT_DIRECTORY_${_CFG} ${ADGENCOV_PY_PKG_DIR})
endforeach()

message(STATUS "adgencov: Python module -> ${ADGENCOV_PY_PKG_DIR}/_core")
