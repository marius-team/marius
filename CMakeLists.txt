cmake_minimum_required(VERSION 3.12.2)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0048 NEW)
project(marius VERSION 0.1 LANGUAGES CXX)

include(FindPackageHandleStandardArgs)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(OPENMP_LIBRARIES "/usr/local/Cellar/llvm/11.0.0/lib")
set(OPENMP_INCLUDES "/usr/local/Cellar/llvm/11.0.0/include")

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
        message(FATAL_ERROR "Clang version must be at least 11!")
    endif()
    set(CLANG TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        message(FATAL_ERROR "GCC version must be at least 9.0!")
    endif()
    set(GCC TRUE)
else ()
    message(FATAL_ERROR "Unknown compiler")
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
endif ()

if(NOT IS_ABSOLUTE ${MARIUS_TORCH_DIR})
    set(MARIUS_TORCH_DIR "${CMAKE_CURRENT_BINARY_DIR}/${MARIUS_TORCH_DIR}")
endif()

set(Torch_DIR ${MARIUS_TORCH_DIR}/share/cmake/Torch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(${USE_CUDA})
    add_definitions(-DMARIUS_CUDA=${USE_CUDA})
endif()

# Find torch location
execute_process(
        COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

execute_process(
        COMMAND python3 -c "import torch; print(torch.__version__, end='')"
        OUTPUT_VARIABLE TorchVersion
)
message(STATUS "Torch Version: ${TorchVersion}")

# Add the cmake folder so the FindSphinx module is found

set(MARIUS_CPP_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/cpp)
set(CMAKE_MODULE_PATH "${MARIUS_CPP_SOURCE}/cmake" ${CMAKE_MODULE_PATH})
set(project_INCLUDE_DIR ${MARIUS_CPP_SOURCE}/include)
set(project_SOURCE_DIR ${MARIUS_CPP_SOURCE}/src)
set(project_TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/test)
set(project_DOCS_DIR ${CMAKE_CURRENT_LIST_DIR}/docs)
set(project_BINDINGS_DIR ${MARIUS_CPP_SOURCE}/python_bindings)
set(project_THIRD_PARTY_DIR ${MARIUS_CPP_SOURCE}/third_party)


set(project_WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DMARIUS_BASE_DIRECTORY="${CMAKE_CURRENT_LIST_DIR}")
add_definitions(-DMARIUS_TEST_DIRECTORY="${project_TEST_DIR}")

if (EXISTS ${project_INCLUDE_DIR})
    file(GLOB_RECURSE project_HEADERS ${project_HEADERS} ${project_INCLUDE_DIR}/*.h)
endif ()
if (EXISTS ${project_SOURCE_DIR})
    file(GLOB_RECURSE project_SOURCES ${project_SOURCES} ${project_SOURCE_DIR}/*.cpp)
endif ()
if (EXISTS ${project_BINDINGS_DIR})
    file(GLOB_RECURSE project_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/*.cpp)
endif ()

add_subdirectory(${project_THIRD_PARTY_DIR})
set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)
include_directories(${project_INCLUDE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${project_THIRD_PARTY_DIR}/inih)
add_library(${PROJECT_NAME} SHARED ${project_SOURCES} ${project_HEADERS})

if (APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" CACHE INTERNAL "OpenMP flags for Xcode toolchain.")
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE INTERNAL "OpenMP lib name for Xcode toolchain.")
    set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib" CACHE INTERNAL "OpenMP lib name for Xcode toolchain.")
endif()


# OpenMP (optional)
option(WITH_OPENMP "Build with parallelization using OpenMP" TRUE)
if(WITH_OPENMP)
    find_package(OpenMP)
endif()
if(OpenMP_FOUND)
    string(APPEND CMAKE_C_FLAGS " ${OpenMP_C_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " ${OpenMP_CXX_FLAGS}")
    if(${CMAKE_VERSION} VERSION_LESS "3.7")
        message(STATUS "Found OpenMP")
    else()
        # We could use OpenMP_CXX_VERSION starting from CMake 3.9, but this value is only available on first run of CMake (see https://gitlab.kitware.com/cmake/cmake/issues/19150),
        # so we use always OpenMP_CXX_SPEC_DATE, which is available since CMake 3.7.
        message(STATUS "Found OpenMP, spec date ${OpenMP_CXX_SPEC_DATE}")
    endif()
    if(MSVC)
        if(MSVC_VERSION EQUAL 1900)
            set(OPENMP_DLL VCOMP140)
        elseif(MSVC_VERSION MATCHES "^191[0-9]$")
            set(OPENMP_DLL VCOMP140)
        elseif(MSVC_VERSION MATCHES "^192[0-9]$")
            set(OPENMP_DLL VCOMP140)
        endif()
        if(OPENMP_DLL)
            string(APPEND CMAKE_SHARED_LINKER_FLAGS_DEBUG " /DELAYLOAD:${OPENMP_DLL}D.dll")
            string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELEASE " /DELAYLOAD:${OPENMP_DLL}.dll")
        else()
            message(WARNING "Delay loading flag for OpenMP DLL is invalid.")
        endif()
    endif()
else()
    message(STATUS "Not found OpenMP")
endif()

if(OpenMP_FOUND)
    add_definitions(-DMARIUS_OMP=1)
endif()

find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} spdlog)
target_link_libraries(${PROJECT_NAME} cxxopts)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${project_HEADERS}")

if (EXISTS ${project_INCLUDE_DIR})
    target_include_directories(${PROJECT_NAME} PUBLIC ${project_INCLUDE_DIR})
endif ()
if (EXISTS ${project_SOURCE_DIR})
    target_include_directories(${PROJECT_NAME} PRIVATE ${project_SOURCE_DIR})
endif ()

target_compile_options(${PROJECT_NAME}
        PRIVATE -fomit-frame-pointer
        PRIVATE -O$<$<CONFIG:Debug>:0 -g3>$<$<CONFIG:Release>:3>
        )

IF(CMAKE_BUILD_TYPE MATCHES Debug AND MARIUS_USE_ASAN)
    message("Using ASAN")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address  -fsanitize=leak")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS}  -fsanitize=address -fsanitize=leak")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug AND MARIUS_USE_ASAN)


IF(BUILD_DOCS)
    add_subdirectory(${project_DOCS_DIR})
ENDIF()

if (EXISTS ${project_TEST_DIR})
    enable_testing()
    add_subdirectory(${project_TEST_DIR})
endif ()

add_executable(marius_train ${project_SOURCE_DIR}/marius.cpp)
add_executable(marius_eval ${project_SOURCE_DIR}/marius.cpp)
target_link_libraries(marius_train ${PROJECT_NAME})
target_link_libraries(marius_eval ${PROJECT_NAME})

pybind11_add_module(_pymarius ${project_SOURCES} ${project_BINDINGS})

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
target_link_libraries(_pymarius PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
target_include_directories(_pymarius PRIVATE ${project_INCLUDE_DIR})