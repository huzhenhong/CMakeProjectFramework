# ----------------------------------------
# 平台相关的库目录配置
# ----------------------------------------
if (WIN32)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-win-64)
    else()
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-win-32)
    endif()
elseif (APPLE)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-macos-64)
    else()
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-macos-32)
    endif()
elseif (UNIX)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-linux-64)
    else()
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-linux-32)
    endif()
endif()

# ----------------------------------------
# 全局编译选项
# ----------------------------------------
if (MSVC)
    add_compile_options(/source-charset:utf-8)

    # Debug / Release / RelWithDebInfo 区分
    add_compile_options("$<$<CONFIG:Debug>:/RTCc>")
    add_compile_options("$<$<CONFIG:Release>:/O2;/GL;/GF;/GS->")
    add_compile_options("$<$<CONFIG:RelWithDebInfo>:/Od>")

    add_compile_options(/W4 /WX /MP /wd4251 /wd4592 /wd4201 /wd4127) # /W4 /WX 严格警告, /MP 多核编译

    add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wunused -Wreorder -Wignored-qualifiers
                        -Wmissing-braces -Wreturn-type -Wswitch -Wswitch-default
                        -Wuninitialized -Wmissing-field-initializers)

    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8)
        add_compile_options(-Wpedantic)
    endif()

    if (OPTION_COVERAGE_ENABLED)
        add_compile_options(-fprofile-arcs -ftest-coverage)
    endif()

    # add_definitions(-DUSE_X64=$<BOOL:${USE_X64}>)

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    # add_definitions(-DUSE_X64=$<BOOL:${USE_X64}>)
endif()

# ----------------------------------------
# CUDA 编译选项
# ----------------------------------------
if (CMAKE_CUDA_COMPILER)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        # $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall>
    )

    # Device code（GPU）编译选项
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

    # Host code（CPU）编译选项通过 -Xcompiler 传给宿主编译器
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wpedantic")


    message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
endif()
