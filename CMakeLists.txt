cmake_minimum_required(VERSION 3.12.2)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0048 NEW)

project(marius VERSION 0.1 LANGUAGES CXX)

include(FindPackageHandleStandardArgs)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_VISIBILITY_PRESET default)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
        message(FATAL_ERROR "Clang version must be at least 11!")
    endif()
    set(CLANG TRUE)
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        message(FATAL_ERROR "GCC version must be at least 7.0!")
    endif()
    set(GCC TRUE)
else ()
    message(FATAL_ERROR "Unknown compiler")
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
endif ()

if(${USE_CUDA})
    add_definitions(-DMARIUS_CUDA=${USE_CUDA})
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
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
set(project_CUDA_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src/cuda/include)
set(project_CUDA_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src/cuda/src)
set(project_CUDA_THIRD_PARTY_DIR ${CMAKE_CURRENT_LIST_DIR}/src/cuda/third_party)
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

if(${USE_CUDA})
    if (EXISTS ${project_CUDA_INCLUDE_DIR})
        file(GLOB_RECURSE project_CUDA_HEADERS ${project_CUDA_INCLUDE_DIR} ${project_CUDA_INCLUDE_DIR}/*.cuh)
    endif ()
    if (EXISTS ${project_CUDA_SOURCE_DIR})
        file(GLOB_RECURSE project_CUDA_SOURCES ${project_CUDA_SOURCE_DIR} ${project_CUDA_SOURCE_DIR}/*.cu)
    endif ()

    if (EXISTS ${project_CUDA_THIRD_PARTY_DIR})
        file(GLOB_RECURSE project_CUDA_THIRD_PARTY_HEADERS ${project_CUDA_THIRD_PARTY_DIR} ${project_CUDA_THIRD_PARTY_DIR}/*.cuh ${project_CUDA_THIRD_PARTY_DIR}/*.h)
    endif ()
    if (EXISTS ${project_CUDA_THIRD_PARTY_DIR})
        file(GLOB_RECURSE project_CUDA_THIRD_PARTY_SOURCES ${project_CUDA_THIRD_PARTY_DIR} ${project_CUDA_THIRD_PARTY_DIR}/*.cu ${project_CUDA_THIRD_PARTY_DIR}/*.cpp)
    endif ()
endif ()

message(STATUS "project_CUDA_THIRD_PARTY_HEADERS ${project_CUDA_THIRD_PARTY_HEADERS}")
message(STATUS "project_CUDA_THIRD_PARTY_SOURCES ${project_CUDA_THIRD_PARTY_SOURCES}")

find_package(Python3 COMPONENTS Development Interpreter REQUIRED)
find_package(Torch REQUIRED)

execute_process(
        COMMAND python3 -c "import torch; print(torch._C._PYBIND11_COMPILER_TYPE, end='')"
        OUTPUT_VARIABLE _PYBIND11_COMPILER_TYPE
)
execute_process(
        COMMAND python3 -c "import torch; print(torch._C._PYBIND11_STDLIB, end='')"
        OUTPUT_VARIABLE _PYBIND11_STDLIB
)
execute_process(
        COMMAND python3 -c "import torch; print(torch._C._PYBIND11_BUILD_ABI, end='')"
        OUTPUT_VARIABLE _PYBIND11_BUILD_ABI
)

message(STATUS "PYBIND11_COMPILER_TYPE:" ${_PYBIND11_COMPILER_TYPE})
message(STATUS "PYBIND11_STDLIB:" ${_PYBIND11_STDLIB})
message(STATUS "PYBIND11_BUILD_ABI:" ${_PYBIND11_BUILD_ABI})

add_compile_definitions(PYBIND11_COMPILER_TYPE="${_PYBIND11_COMPILER_TYPE}" PYBIND11_STDLIB="${_PYBIND11_STDLIB}" PYBIND11_BUILD_ABI="${_PYBIND11_BUILD_ABI}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

message(STATUS "Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS}")
add_subdirectory(${project_THIRD_PARTY_DIR})
set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${project_INCLUDE_DIR})
include_directories(${project_CUDA_INCLUDE_DIR})
include_directories(${project_CUDA_THIRD_PARTY_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${project_THIRD_PARTY_DIR}/parallel-hashmap/)
include_directories(${project_BINDINGS})

add_library(${PROJECT_NAME}
            SHARED
            ${project_SOURCES}
            ${project_HEADERS}
            ${project_CUDA_HEADERS}
            ${project_CUDA_SOURCES}
            ${project_CUDA_THIRD_PARTY_HEADERS}
            ${project_CUDA_THIRD_PARTY_SOURCES})

if(NOT APPLE)
    target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} spdlog)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${project_HEADERS}")
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(${USE_CUDA})
    set(NVCC_FLAGS "{NVCC_FLAGS} --expt-relaxed-constexpr")
endif()

if(${USE_OMP})
    add_definitions(-DMARIUS_OMP=${USE_OMP})
    if(APPLE)
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY omp)
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY omp)
        endif()
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp")
    endif()
    find_package(OpenMP REQUIRED)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

if (EXISTS ${project_INCLUDE_DIR})
    target_include_directories(${PROJECT_NAME} PUBLIC ${project_INCLUDE_DIR})
endif ()
if (EXISTS ${project_SOURCE_DIR})
    target_include_directories(${PROJECT_NAME} PRIVATE ${project_SOURCE_DIR})
endif ()

IF(CMAKE_BUILD_TYPE MATCHES Debug AND MARIUS_USE_TSAN)
    message("Using address sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug AND MARIUS_USE_TSAN)

IF(CMAKE_BUILD_TYPE MATCHES Debug AND MARIUS_USE_ASAN)
    message("Using thread sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak")
    set(CMAKE_MODULE_LINKER_FLAGS "{$CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
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

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

file(GLOB_RECURSE CONFIG_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/configuration/*.cpp)
pybind11_add_module(_config ${CONFIG_BINDINGS})
target_link_libraries(_config PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE DATA_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/data/*.cpp)
pybind11_add_module(_data ${DATA_BINDINGS})
target_link_libraries(_data PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE NN_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/nn/*.cpp)
pybind11_add_module(_nn ${NN_BINDINGS})
target_link_libraries(_nn PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE MANAGER_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/manager/*.cpp)
pybind11_add_module(_manager ${MANAGER_BINDINGS})
target_link_libraries(_manager PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE PIPELINE_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/pipeline/*.cpp)
pybind11_add_module(_pipeline ${PIPELINE_BINDINGS})
target_link_libraries(_pipeline PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE REPORT_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/reporting/*.cpp)
pybind11_add_module(_report ${REPORT_BINDINGS})
target_link_libraries(_report PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

file(GLOB_RECURSE STORAGE_BINDINGS ${project_BINDINGS} ${project_BINDINGS_DIR}/storage/*.cpp)
pybind11_add_module(_storage ${STORAGE_BINDINGS})
target_link_libraries(_storage PRIVATE ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

add_custom_target(bindings)
add_dependencies(bindings _config _data _manager _nn _pipeline _report _storage)