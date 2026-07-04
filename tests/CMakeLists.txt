include(Catch)

add_executable(adgencov_tests
  test_projection.cpp
  test_shrink.cpp)

# golden_shrink.hpp is generated next to the test sources.
target_include_directories(adgencov_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(adgencov_tests PRIVATE
  adgencov::adgencov
  Catch2::Catch2WithMain)

catch_discover_tests(adgencov_tests)
