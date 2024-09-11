cmake_minimum_required(VERSION 3.20)
project(SimSIMDExample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


include(FetchContent)

# Declare and fetch SimSIMD
FetchContent_Declare(
    simsimd
    GIT_REPOSITORY https://github.com/ashvardanian/simsimd.git
    GIT_SHALLOW TRUE
    GIT_TAG main
)
FetchContent_MakeAvailable(simsimd)

# Find libnuma
find_package(PkgConfig REQUIRED)
pkg_check_modules(NUMA REQUIRED numa)

# Add your executable
add_executable(main main.cpp)

# Link SimSIMD and libnuma to your executable
target_link_libraries(main PRIVATE simsimd ${NUMA_LIBRARIES})

# Include SimSIMD and libnuma headers
target_include_directories(main PRIVATE 
    ${simsimd_SOURCE_DIR}/include
    ${NUMA_INCLUDE_DIRS}
)

# Add compile options for libnuma
target_compile_options(main PRIVATE ${NUMA_CFLAGS_OTHER})

# Add preprocessor definitions to handle missing types
target_compile_definitions(main PRIVATE
    SIMSIMD_NATIVE_F16=0
    SIMSIMD_NATIVE_BF16=0
)