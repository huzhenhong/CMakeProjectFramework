/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 15:09:42
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 15:36:17
 * FilePath     : \\CMakeProjectFramework\\src\\demo\\ExportAbstractClass\\AbstractClassInterface.cpp
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "AbstractClassInterface.h"
#include "ExportClass.h"


IExportAbstractClass* CreateExportAbstractClass()
{
    return new ExportClass();
}


LIBRARY_API std::shared_ptr<IExportAbstractClass> CreateExportAbstractClassPtr()
{
    return std::make_shared<ExportClass>();
}