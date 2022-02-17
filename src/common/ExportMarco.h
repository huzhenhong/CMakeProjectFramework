/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 14:14:23
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 16:15:25
 * FilePath     : \\CMakeProjectFramework\\src\\common\\ExportMarco1.h
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once

#if (defined(WIN32) || defined(WIN64))
    // #define WIN
    #ifdef LIBRARY_EXPORT
        #define LIBRARY_API __declspec(dllexport)
    #else
        #define LIBRARY_API __declspec(dllimport)
    #endif  // LIBRARY_EXPORT
#else       // NOT define win32 or win64
    #define LIBRARY_API
#endif

#define EXTERN_C extern "C"
