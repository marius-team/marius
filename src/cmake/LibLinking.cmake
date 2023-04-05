target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} spdlog)

message(STATUS "Linking: ${TORCH_LIBRARIES} spdlog")

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${project_HEADERS}")
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

if(${USE_OMP})
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()
