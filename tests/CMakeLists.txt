include(Catch)

add_executable(adgencov_tests
  test_projection.cpp)

target_link_libraries(adgencov_tests PRIVATE
  adgencov::adgencov
  Catch2::Catch2WithMain)

catch_discover_tests(adgencov_tests)
