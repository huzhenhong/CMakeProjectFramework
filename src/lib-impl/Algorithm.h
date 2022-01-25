/*************************************************************************************
 * Description  : 
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 14:27:20
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-01-20 14:28:29
 * FilePath     : \\CMakeProjectFrame\\src\\Algorithm\\Algorithm.h
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once
#include "IAlgorithm.h"

class Algorithm : public IAlgorithm
{
  public:
    Algorithm();
    ~Algorithm();

    int Sum(int a, int b) override;
};
