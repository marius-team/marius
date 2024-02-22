file(GLOB TEST_HEADERS *.h)
file(GLOB SRCS *.cpp)

add_library(${TEST_LIB}
        SHARED
        ${TEST_HEADERS}
        ${SRCS})
target_link_libraries(${TEST_LIB} ${PROJECT_NAME})
target_link_libraries(${TEST_LIB} gtest)
target_link_libraries(${TEST_LIB} gtest_main)

ADD_EXECUTABLE(unit ${SRCS})

TARGET_LINK_LIBRARIES(unit ${TEST_LIB} ${PROJECT_NAME} gtest gtest_main)

add_test(NAME unit COMMAND unit WORKING_DIRECTORY ${project_WORKING_DIR})