// src/my_cuda_lib.cu

#include <cuda_runtime.h>
#include "cuda_algo.h"

// CUDA 核函数：执行 Y = a * X + Y
__global__ void saxpy_kernel(float a, const float* x, float* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        y[i] = a * x[i] + y[i];
    }
}

// C ABI 接口：这是我们导出的稳定C函数
// 使用 extern "C" 来防止 C++ 的名字修饰（name mangling）
extern "C" cudaError_t saxpy_c(float a, const float* x, float* y, int n)
{
    if (n <= 0)
    {
        return cudaSuccess;
    }

    // 计算执行配置
    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 启动核函数
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, x, y, n);

    // cudaGetLastError() 用于捕获异步的核函数启动错误
    return cudaGetLastError();
}