/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 15:09:42
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-02-17 16:17:30
 * FilePath     : \\CMakeProjectFramework\\src\\demo\\ExportC\\CInterface.cpp
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "CInterface.h"
#include "tracy/Tracy.hpp"
#include <unistd.h>

int Sum(int a, int b)
{
    TracyMessageL("Hello");
    sleep(1000);
    return a + b;
}
