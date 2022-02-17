/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 13:58:58
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 15:50:42
 * FilePath     : \\CMakeProjectFramework\\test\\TestClass.cpp
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "../src/ExportClass/ClassInterface.h"
#include <iostream>


int main(int argc, char* argv[])
{
    ClassInterface impl;

    std::cout << "1 + 2 = " << impl.Sum(1, 2) << std::endl;

    return 0;
}