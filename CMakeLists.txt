cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project("CMakeProjectFramework" 
    VERSION 1.0.0.0
    LANGUAGES C CXX
    DESCRIPTION "something about this project"
    HOMEPAGE_URL "project site"
    )

message("\n================= Start configure project [${PROJECT_NAME}] ==================")

# 设置默认输出路径
set(CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pdb)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


option(BUILD_WITH_EXAMPLE "option for build example" ON)
option(BUILD_WITH_TEST "option for build test" ON)
option(BUILD_WITH_CUDA "option for build cuda" ON)
option(BUILD_WITH_CATCH2 "option for build catch2" ON)
option(BUILD_WITH_TRACY "option for build tracy" ON)
option(BUILD_WITH_SPDLOG "option for build spdlog" ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common_func.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain.cmake)
# include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_link_options.cmake)

add_subdirectory(src)

if(BUILD_WITH_EXAMPLE)
    add_subdirectory(example)
endif()

# if(BUILD_WITH_TEST)
#     add_subdirectory(test)
# endif()

if(BUILD_WITH_CUDA)
    add_compile_definitions(BUILD_WITH_CUDA)
endif()

# if(BUILD_WITH_TRACY)
#     add_compile_definitions(BUILD_WITH_TRACY)

#     add_library(TracyClient ${CMAKE_CURRENT_SOURCE_DIR}/ext/tracy/public/TracyClient.cpp)
#     target_include_directories(TracyClient PUBLIC ext/tracy)
#     target_link_libraries(example PRIVATE TracyClient)
# endif()

# --------------------
# spdlog
# --------------------
# if(BUILD_WITH_SPDLOG)
#     if(NOT TARGET spdlog)
#         add_subdirectory(ext/spdlog)
#     endif()
#     target_link_libraries(example PRIVATE spdlog::spdlog)
# endif()

# --------------------
# Catch2
# --------------------
if(BUILD_WITH_CATCH2)
    if(NOT TARGET Catch2::Catch2WithMain)
        add_subdirectory(ext/catch2)
    endif()

    # add_executable(my_tests tests/test_main.cpp)
    # target_link_libraries(my_tests PRIVATE Catch2::Catch2WithMain)

    # # 可选：启用 Catch2 的 CTest 集成
    # include(CTest)
    # include(Catch)
    # catch_discover_tests(my_tests)
endif()

# --------------------
# Tracy
# --------------------
if(BUILD_WITH_TRACY)
    # 方案 1：只构建 client (推荐)
    add_library(TracyClient ${CMAKE_CURRENT_SOURCE_DIR}/ext/tracy/public/TracyClient.cpp)
    target_include_directories(TracyClient PUBLIC ext/tracy)

    target_link_libraries(example PRIVATE TracyClient)

    # 方案 2（可选）：用 Tracy 自带 CMake，但要加保护
    # if(NOT TARGET TracyClient)
    #     set(TRACY_ENABLE ON CACHE BOOL "Enable Tracy profiling" FORCE)
    #     add_subdirectory(ext/tracy)
    # endif()
    # target_link_libraries(example PRIVATE TracyClient)
endif()

message("================== Stop configure project [${PROJECT_NAME}] ==================\n")