/*************************************************************************************
 * Description  :
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 13:58:58
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-04-07 13:56:31
 * FilePath     : \\CMakeProjectFramework\\test\\TestC.cpp
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "../src/ExportC/CInterface.h"
#include <stdio.h>


int main(int argc, char* argv[])
{
    printf("1 + 2 = %d\n", Sum(1, 2));

    return 0;
}