/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 14:14:23
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 15:23:27
 * FilePath     : \\CMakeProjectFramework\\src\\demo\\ExportAbstractClass\\impl.h
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once
#include "../common/ExportMarco.h"
#include "IExportAbstractClass.h"
#include <memory>


LIBRARY_API IExportAbstractClass* CreateExportAbstractClass();

LIBRARY_API std::shared_ptr<IExportAbstractClass> CreateExportAbstractClassPtr();
