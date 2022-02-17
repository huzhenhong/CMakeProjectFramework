/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 14:27:20
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 15:13:41
 * FilePath     : \\CMakeProjectFramework\\src\\demo\\ExportAbstractClass\\ExportClass.h
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once
#include "IExportAbstractClass.h"


class ExportClass : public IExportAbstractClass
{
  public:
    ExportClass();
    ~ExportClass();

    int Sum(int a, int b) override;
};
