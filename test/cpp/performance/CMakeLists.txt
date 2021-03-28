file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(performance ${SRCS})

TARGET_LINK_LIBRARIES(performance
        marius
        libgtest
        )

add_test(NAME performance COMMAND performance)