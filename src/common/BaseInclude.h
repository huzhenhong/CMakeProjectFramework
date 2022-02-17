/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2021-06-19 04:02:57
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 14:26:57
 * FilePath     : \\CMakeProjectFramework\\src\\common\\BaseInclude.h
 * Copyright (C) 2021 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once

#ifndef __linux__
    #include <Windows.h>
#else
    #include <dlfcn.h>
    #include <sys/syscall.h>
    #include <unistd.h>
#endif  // !__linux__

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && \
    defined(__has_include)
    #if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
        #define GHC_USE_STD_FS
        #include <filesystem>
namespace fs = std::filesystem;
    #endif
#endif
#ifndef GHC_USE_STD_FS
    #include "filesystem.hpp"
namespace fs = ghc::filesystem;
#endif
