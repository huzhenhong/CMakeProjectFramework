/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 14:14:23
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 15:13:36
 * FilePath     : \\CMakeProjectFramework\\src\\demo\\ExportAbstractClass\\IExportAbstractClass.h
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once


class IExportAbstractClass
{
  public:
    IExportAbstractClass()          = default;
    virtual ~IExportAbstractClass() = default;

    virtual int Sum(int a, int b) = 0;
};
