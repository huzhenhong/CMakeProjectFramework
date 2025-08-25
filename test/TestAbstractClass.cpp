/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 13:58:58
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 16:23:56
 * FilePath     : \\CMakeProjectFramework\\test\\TestAbstractClass.cpp
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "../src/ExportAbstractClass/AbstractClassInterface.h"
#include <iostream>


int main(int argc, char* argv[])
{
    auto exportAbstractClassPtr  = CreateExportAbstractClass();
    auto exportAbstractClassSPtr = CreateExportAbstractClassPtr();

    std::cout << "1 + 2 = " << exportAbstractClassPtr->Sum(1, 2) << std::endl;
    std::cout << "1 + 2 = " << exportAbstractClassSPtr->Sum(1, 2) << std::endl;

    return 0;
}