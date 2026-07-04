# The adgencov command-line tool.
add_executable(adgencov_cli main.cpp)
set_target_properties(adgencov_cli PROPERTIES OUTPUT_NAME adgencov)
target_link_libraries(adgencov_cli PRIVATE adgencov::adgencov)

install(TARGETS adgencov_cli RUNTIME DESTINATION bin)
