file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(integration ${SRCS})

TARGET_LINK_LIBRARIES(integration
        ${PROJECT_NAME}
        gtest gtest_main
        )

add_test(NAME integration COMMAND integration WORKING_DIRECTORY ${project_WORKING_DIR})