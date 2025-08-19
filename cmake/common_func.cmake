function(print_build_environment_summary)
    message(STATUS "\n================= Project Build Summary =================")

    # --- 主机信息 ---
    message(STATUS "- System name: ${CMAKE_SYSTEM_NAME}")
    message(STATUS "- Host processor: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
    message(STATUS "- Pointer size: ${CMAKE_SIZEOF_VOID_P} bytes")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(STATUS "- Processor is 64-bit")
    else()
        message(STATUS "- Processor is 32-bit")
    endif()

    # 基础信息
    message(STATUS "CMake Version       : ${CMAKE_VERSION}")
    message(STATUS "System              : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")
    message(STATUS "Processor           : ${CMAKE_SYSTEM_PROCESSOR}")
    message(STATUS "Build Type          : ${CMAKE_BUILD_TYPE}")

    # 架构信息
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(STATUS "Target Architecture : 64-bit")
    else()
        message(STATUS "Target Architecture : 32-bit")
    endif()

    # C 语言信息
    if(CMAKE_C_COMPILER)
        message(STATUS "C Compiler          : ${CMAKE_C_COMPILER}")
        message(STATUS "C Compiler ID       : ${CMAKE_C_COMPILER_ID}")
        message(STATUS "C Compiler Version  : ${CMAKE_C_COMPILER_VERSION}")
        if(CMAKE_C_STANDARD)
            message(STATUS "C Standard          : ${CMAKE_C_STANDARD}")
        endif()
    endif()

    # C++ 语言信息
    if(CMAKE_CXX_COMPILER)
        message(STATUS "C++ Compiler        : ${CMAKE_CXX_COMPILER}")
        message(STATUS "C++ Compiler ID     : ${CMAKE_CXX_COMPILER_ID}")
        message(STATUS "C++ Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
        if(CMAKE_CXX_STANDARD)
            message(STATUS "C++ Standard        : ${CMAKE_CXX_STANDARD}")
        endif()
    endif()

    # CUDA 信息
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA Compiler       : ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CUDA Compiler ID    : ${CMAKE_CUDA_COMPILER_ID}")
        message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
        if(CMAKE_CUDA_STANDARD)
            message(STATUS "CUDA Standard       : ${CMAKE_CUDA_STANDARD}")
        endif()
    endif()

    # 编译选项
    message(STATUS "C Flags             : ${CMAKE_C_FLAGS}")
    message(STATUS "C++ Flags           : ${CMAKE_CXX_FLAGS}")
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA Flags          : ${CMAKE_CUDA_FLAGS}")
    endif()

    # 链接选项
    message(STATUS "Executable Linker   : ${CMAKE_EXE_LINKER_FLAGS}")
    message(STATUS "Shared Linker       : ${CMAKE_SHARED_LINKER_FLAGS}")
    message(STATUS "Module Linker       : ${CMAKE_MODULE_LINKER_FLAGS}")

    message(STATUS "=========================================================\n")
endfunction()



# --- 链接库目录封装 ---
function(configure_target_thirdparty_link_dirs target)
    # 生成表达式选择路径
    target_link_directories(${target} PRIVATE
        $<$<AND:$<PLATFORM_ID:Windows>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:${CMAKE_SOURCE_DIR}/3rdparty/lib-win-64>
        $<$<AND:$<PLATFORM_ID:Windows>,$<NOT:$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>>:${CMAKE_SOURCE_DIR}/3rdparty/lib-win-32>

        $<$<AND:$<PLATFORM_ID:Darwin>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:${CMAKE_SOURCE_DIR}/3rdparty/lib-macos-64>
        $<$<AND:$<PLATFORM_ID:Darwin>,$<NOT:$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>>:${CMAKE_SOURCE_DIR}/3rdparty/lib-macos-32>

        $<$<AND:$<PLATFORM_ID:Linux>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:${CMAKE_SOURCE_DIR}/3rdparty/lib-linux-64>
        $<$<AND:$<PLATFORM_ID:Linux>,$<NOT:$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>>:${CMAKE_SOURCE_DIR}/3rdparty/lib-linux-32>
    )

    # 对 iOS / Android 提示警告
    if(IOS)
        message(WARNING "iOS link directories not specified")
    elseif(ANDROID)
        message(WARNING "Android link directories not specified")
    endif()

endfunction()


# --- 编译选项封装函数 ---
function(configure_target_compile_options target)

    target_compile_options(${target} PRIVATE
        # MSVC only
        $<$<C_COMPILER_ID:MSVC>:/source-charset:utf-8>
        $<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>
        $<$<AND:$<C_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/RTCc>
        $<$<AND:$<C_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /GL /GF /GS->
        $<$<AND:$<C_COMPILER_ID:MSVC>,$<CONFIG:RelWithDebInfo>>:/Od>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /MP /wd4251 /wd4592 /wd4201 /wd4127>

        # GNU only
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wunused -Wreorder -Wignored-qualifiers -Wmissing-braces -Wreturn-type -Wswitch -Wswitch-default -Wuninitialized -Wmissing-field-initializers>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,4.8>>:-Wpedantic>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<BOOL:${OPTION_COVERAGE_ENABLED}>>:-fprofile-arcs -ftest-coverage>

        # AppleClang only
        $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Wpedantic>

        # CUDA only
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xcompiler -Wall>
    )

    target_compile_definitions(${target} PRIVATE
        # MSVC definitions
        $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_WARNINGS NOMINMAX>

        # GCC / AppleClang definitions
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:AppleClang>>:USE_X64=$<BOOL:${USE_X64}>>
    )

endfunction()

