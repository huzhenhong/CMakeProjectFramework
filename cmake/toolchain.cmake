if(EXISTS /cambricon)
    set(tools /opt/cambricon/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/)
    set(CMAKE_C_COMPILER ${tools}/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER ${tools}/bin/aarch64-linux-gnu-g++)

elseif(EXISTS /opt/rh/devtoolset-7/root/usr/bin/gcc)
    set(tools /opt/rh/devtoolset-7/root/usr/)
    set(CMAKE_C_COMPILER ${tools}/bin/gcc)
    set(CMAKE_CXX_COMPILER ${tools}/bin/g++)

elseif(EXISTS /opt/sophon)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
    if(DEFINED SOPHON_ARCH AND SOPHON_ARCH STREQUAL "x86_64")
        # pcie 交叉编译，由于都是在容器里交叉编译，无法区分cpu，只能传参数
        set(CMAKE_C_COMPILER /usr/bin/x86_64-linux-gnu-gcc)
        set(CMAKE_ASM_COMPILER /usr/bin/x86_64-linux-gnu-gcc)
        set(CMAKE_CXX_COMPILER /usr/bin/x86_64-linux-gnu-g++)
    else()
        # 算能小盒子交叉编译
        set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
        set(CMAKE_ASM_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
        set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
    endif()
endif()