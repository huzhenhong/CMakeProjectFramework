/*************************************************************************************
 * Description  : 
 * Version      : 1.0
 * Author       : huzhenhong
 * Date         : 2022-01-20 14:14:23
 * LastEditors  : huzhenhong
 * LastEditTime : 2022-01-20 15:09:33
 * FilePath     : \\CMakeProjectFrame\\src\\Algorithm\\IAlgorithm.h
 * Copyright (C) 2022 huzhenhong. All rights reserved.
 *************************************************************************************/
#pragma once
#include <memory>


class IAlgorithm
{
  public:
    IAlgorithm()          = default;
    virtual ~IAlgorithm() = default;

    virtual int Sum(int a, int b) = 0;
};

using IAlgorithmPtr = std::shared_ptr<IAlgorithm>;

IAlgorithmPtr CreateAlgorithm();