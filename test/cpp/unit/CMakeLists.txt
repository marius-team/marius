file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(unit ${SRCS})

TARGET_LINK_LIBRARIES(unit
        ${PROJECT_NAME}
        gtest gtest_main
        )

add_test(NAME unit COMMAND unit)