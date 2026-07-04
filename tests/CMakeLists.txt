include(Catch)

add_executable(adgencov_tests
  test_projection.cpp
  test_shrink.cpp
  test_select.cpp
  test_io.cpp
  test_pipeline.cpp)

# golden_*.hpp headers are generated next to the test sources; the tests also
# read fixtures under this directory, so expose its path.
target_include_directories(adgencov_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(adgencov_tests PRIVATE
  ADGENCOV_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(adgencov_tests PRIVATE
  adgencov::adgencov
  Catch2::Catch2WithMain)

catch_discover_tests(adgencov_tests)
