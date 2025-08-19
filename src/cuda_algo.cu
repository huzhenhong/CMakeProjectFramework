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

// C ABI 接口：现在是异步版本
extern "C" cudaError_t saxpy_c_async(float a, const float* x, float* y, int n, cudaStream_t stream)
{
    if (n <= 0)
    {
        return cudaSuccess;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 关键改动：将 stream 参数传递给核函数启动配置（第四个参数）
    // 第三个参数是动态共享内存大小，这里我们不需要，所以是 0
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, x, y, n);

    // 核函数启动是异步的，使用 cudaGetLastError 捕获启动时的错误
    return cudaGetLastError();
}