# Find torch location
execute_process(
        COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)
list(PREPEND CMAKE_PREFIX_PATH ${TorchPath})

## favor conda packages over system packages
#if (EXISTS $ENV{CONDA_PREFIX})
#    set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
#endif ()

execute_process(
        COMMAND python3 -c "import torch; print(torch.__version__, end='')"
        OUTPUT_VARIABLE TorchVersion
)


find_package(Torch REQUIRED)


# find python which is used to build torch

execute_process(
        COMMAND python3 -c "import sys; print(sys.executable, end='')"
        OUTPUT_VARIABLE Python3_EXECUTABLE
)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# print debugging information about python
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
message(STATUS "Python3_VERSION: ${Python3_VERSION}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

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
endif()

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
message(STATUS "Torch Version: ${TorchVersion}")

add_compile_definitions(PYBIND11_COMPILER_TYPE="${_PYBIND11_COMPILER_TYPE}" PYBIND11_STDLIB="${_PYBIND11_STDLIB}" PYBIND11_BUILD_ABI="${_PYBIND11_BUILD_ABI}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(${project_THIRD_PARTY_DIR})
