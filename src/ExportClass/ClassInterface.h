/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 14:27:20
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 15:50:04
 * FilePath     : \\CMakeProjectFramework\\src\\demo\\ExportClass\\ClassInterface.h
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once
#include "../common/ExportMarco.h"


class LIBRARY_API ClassInterface
{
  public:
    ClassInterface();
    ~ClassInterface();

    int Sum(int a, int b);
};
