file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(integration ${SRCS})

TARGET_LINK_LIBRARIES(integration
        marius
        libgtest
        )

add_test(NAME integration COMMAND integration)