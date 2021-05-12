find_package(Threads REQUIRED)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

## Add links to test files below
add_subdirectory(end_to_end)
add_subdirectory(integration)
add_subdirectory(performance)
add_subdirectory(unit)
