message("\n------------------ Start generate : Test ------------------")

aux_source_directory(. TEST_SRC_LIST)

foreach(TEST_SRC ${TEST_SRC_LIST})
    STRING(REGEX REPLACE "^\\./|\\.c[a-zA-Z0-9_]*$" "" TEST_EXE_NAME ${TEST_SRC})
    message(STATUS "添加测试程序:${TEST_EXE_NAME}")
    
    add_executable(${TEST_EXE_NAME} ${TEST_SRC})
    # target_include_directories(${TEST_EXE_NAME} 
    #     PRIVATE
    #         ${CMAKE_CURRENT_SOURCE_DIR}
    #     )

    target_link_directories(${TEST_EXE_NAME}
        PRIVATE
            ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
        )

    target_link_libraries(${TEST_EXE_NAME} 
        PRIVATE 
            ${PROJECT_NAME}
        )

    include(FetchContent)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git 
        GIT_TAG v3.5.4
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/catch2
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        OPTIONS "-DBUILD_EXTRAS=OFF" # 禁止构建额外组件
        )
    FetchContent_MakeAvailable(catch2)

    # target_link_libraries(${Target} PUBLIC Tracy::TracyClient)

endforeach()

message("------------------ Stop generate : Test ------------------\n")
