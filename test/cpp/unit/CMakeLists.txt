file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(unit ${SRCS})

TARGET_LINK_LIBRARIES(unit
        marius
        libgtest
        )

add_test(NAME unit COMMAND unit)