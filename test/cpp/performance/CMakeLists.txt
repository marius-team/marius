file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(performance ${SRCS})

TARGET_LINK_LIBRARIES(performance
        ${PROJECT_NAME}
        gtest gtest_main
        )

add_test(NAME performance COMMAND performance WORKING_DIRECTORY ${project_WORKING_DIR})