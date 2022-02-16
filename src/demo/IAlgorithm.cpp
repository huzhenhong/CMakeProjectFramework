/*************************************************************************************
 * Description  : 
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 15:09:42
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-01-20 15:09:42
 * FilePath     : \\CMakeProjectFrame\\src\\Algorithm\\IAlgorithm.cpp
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#include "IAlgorithm.h"
#include "Algorithm.h"


IAlgorithmPtr CreateAlgorithm()
{
    return std::make_shared<Algorithm>();
}