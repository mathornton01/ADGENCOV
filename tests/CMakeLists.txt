include(Catch)

add_executable(adgencov_tests
  test_projection.cpp
  test_shrink.cpp
  test_select.cpp
  test_io.cpp
  test_pipeline.cpp
  test_clustering.cpp)

# golden_*.hpp headers are generated next to the test sources; the tests also
# read fixtures under this directory, so expose its path.
target_include_directories(adgencov_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(adgencov_tests PRIVATE
  ADGENCOV_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(adgencov_tests PRIVATE
  adgencov::adgencov
  Catch2::Catch2WithMain)

catch_discover_tests(adgencov_tests)

# ---- Python binding parity tests -----------------------------------------
# When the pybind11 module is built, register the pytest parity suite as a
# ctest so `ctest` exercises the Python bindings alongside the C++ tests.
# Requires numpy/scikit-learn/pytest on the chosen interpreter (the golden
# venv, or any env with those packages).
if(ADGENCOV_BUILD_PYTHON)
  add_test(NAME python_bindings_parity
    COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/test_bindings.py)
  # The pure-Python package lives in python/; the compiled _core lands there.
  set_tests_properties(python_bindings_parity PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python"
    DEPENDS adgencov_core)

  # GEO ingestion (Phase B) — parses a committed series-matrix fixture and runs
  # the GEO->recommender path fully offline (no network, no GEOparse needed).
  # Requires numpy + pandas on the chosen interpreter.
  add_test(NAME geo_ingestion
    COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/test_geo.py)
  set_tests_properties(geo_ingestion PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python"
    DEPENDS adgencov_core)
endif()
