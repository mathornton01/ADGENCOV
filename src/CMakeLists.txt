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
