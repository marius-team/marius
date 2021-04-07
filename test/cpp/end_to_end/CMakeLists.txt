file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(end_to_end ${SRCS})

TARGET_LINK_LIBRARIES(end_to_end
        ${PROJECT_NAME}
        gtest gtest_main
        )

add_test(NAME end_to_end COMMAND end_to_end)