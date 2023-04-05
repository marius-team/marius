include_directories(${Python3_INCLUDE_DIRS})
include_directories(${project_INCLUDE_DIR})
include_directories(${project_CUDA_INCLUDE_DIR})
include_directories(${project_CUDA_THIRD_PARTY_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${project_THIRD_PARTY_DIR}/parallel-hashmap/)
include_directories(${project_BINDINGS})

file(GLOB_RECURSE project_HEADERS ${project_HEADERS} ${project_INCLUDE_DIR}/*.h)
file(GLOB_RECURSE project_SOURCES ${project_SOURCES} ${project_SOURCE_DIR}/*.cpp)

add_library(${PROJECT_NAME}
        SHARED
        ${project_SOURCES}
        ${project_HEADERS}
        ${project_CUDA_HEADERS}
        ${project_CUDA_SOURCES}
        ${project_CUDA_THIRD_PARTY_HEADERS}
        ${project_CUDA_THIRD_PARTY_SOURCES})

IF(BUILD_DOCS)
    add_subdirectory(${project_DOCS_DIR})
ENDIF()

if (EXISTS ${project_TEST_DIR})
    enable_testing()
    add_subdirectory(${project_TEST_DIR})
endif ()

add_executable(marius_train ${project_SOURCE_DIR}/marius.cpp)
add_executable(marius_eval ${project_SOURCE_DIR}/marius.cpp)
target_link_libraries(marius_train ${PROJECT_NAME} ${Python3_LIBRARIES})
target_link_libraries(marius_eval ${PROJECT_NAME} ${Python3_LIBRARIES})