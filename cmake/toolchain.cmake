
set(CMAKE_C_STANDARD 11)                    # 设置整个项目的默认 C 标准
set(CMAKE_CXX_STANDARD 17)                  # 设置整个项目的默认 C++ 标准
set(CMAKE_CXX_EXTENSIONS OFF)               # 只启用ISO C++标准的编译器标志，而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD_REQUIRED ON)         # 开启防止编译器回退到支持的低版本的C++标准
set(CMAKE_POSITION_INDEPENDENT_CODE ON)     # 默认为开启，代码位置无关，避免使用动态链接库时需要重定位
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)    # 在 Windows 上导出所有库符号
set(CMAKE_MACOSX_RPATH ON)                  # macOS上动态库使用相对路径
set(CMAKE_DEBUG_POSTFIX "_d")


# 平台判断
if (WIN32)
    message(STATUS "Windows platform detected")
    # Windows 下你可能直接用 MSVC，不需要手动设置
    # 但如果要指定 MinGW 或交叉工具链，可以在这里写
    # set(CMAKE_C_COMPILER "gcc.exe")
    # set(CMAKE_CXX_COMPILER "g++.exe")

elseif(APPLE)
    message(STATUS "macOS platform detected")
    # macOS 默认就会用 clang，不一定要手动设置
    # set(CMAKE_C_COMPILER "/usr/bin/clang")
    # set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

elseif(UNIX AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Linux platform detected")

    if(EXISTS /cambricon)
        message(STATUS "Cambricon environment found")
        set(tools /opt/cambricon/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/)
        set(CMAKE_C_COMPILER ${tools}/bin/aarch64-linux-gnu-gcc)
        set(CMAKE_CXX_COMPILER ${tools}/bin/aarch64-linux-gnu-g++)
    elseif(EXISTS /opt/sophon)
        message(STATUS "Sophon environment found")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
        if(DEFINED SOPHON_ARCH AND SOPHON_ARCH STREQUAL "x86_64")
            # pcie 交叉编译
            set(CMAKE_C_COMPILER /usr/bin/x86_64-linux-gnu-gcc)
            set(CMAKE_ASM_COMPILER /usr/bin/x86_64-linux-gnu-gcc)
            set(CMAKE_CXX_COMPILER /usr/bin/x86_64-linux-gnu-g++)
        else()
            # 算能小盒子交叉编译
            set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
            set(CMAKE_ASM_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
            set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
        endif()
    elseif(EXISTS /usr/bin)
        set(CMAKE_C_COMPILER /usr/bin/gcc)
        set(CMAKE_CXX_COMPILER /usr/bin/g++)
        if(BUILD_WITH_CUDA)
            if(EXISTS /usr/local/cuda)
                message(STATUS "CUDA environment found")
                set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
                # 链接 CUDA runtime（通常自动，但显式更清晰）
                find_package(CUDAToolkit REQUIRED)
                enable_language(CUDA)
                # set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)
                set(CMAKE_CUDA_STANDARD 17)
                set(CMAKE_CUDA_STANDARD_REQUIRED ON)
                set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
                set(CMAKE_CUDA_ARCHITECTURES native) # CMake 3.24+ 需要

            else()
                message(STATUS "CUDA environment not found")
            endif()
        endif()
    else()
        message(STATUS "No special toolchain detected, use default compilers")
        message("Default C compiler: ${CMAKE_C_COMPILER}")
        message("Default C++ compiler: ${CMAKE_CXX_COMPILER}")
    endif()

else()
    message(WARNING "Unknown platform: ${CMAKE_SYSTEM_NAME}")
endif()

print_build_environment_summary()
