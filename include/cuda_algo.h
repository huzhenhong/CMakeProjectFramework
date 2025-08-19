#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief 执行 SAXPY 操作 (Y = a * X + Y) 的 C 接口
     * @param a 标量 a
     * @param x 指向设备内存中向量 X 的指针
     * @param y 指向设备内存中向量 Y 的指针
     * @param n 向量的元素个数
     * @return cudaError_t CUDA API 的错误码
     */
    cudaError_t saxpy_c(float a, const float* x, float* y, int n);

#ifdef __cplusplus
}
#endif
