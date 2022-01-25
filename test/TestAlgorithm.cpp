/*************************************************************************************
 * Description  : 
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 13:58:58
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-01-24 17:05:44
 * FilePath     : \\CMakeProjectFrame\\test\\TestAlgorithm.cpp
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "../src/Algorithm/IAlgorithm.h"
#include <iostream>

int main(int argc, char* argv[])
{
    auto algorithmPtr = CreateAlgorithm();

    std::cout << "1 + 2 = " << algorithmPtr->Sum(1, 2) << std::endl;

    return 0;
}