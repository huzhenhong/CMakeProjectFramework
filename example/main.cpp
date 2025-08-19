// example/main.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include "cuda_algo.hpp"  // 只需要包含这个现代化的 C++ 头文件


int main()
{
    try
    {
        const int          N = 1 << 20;  // 约一百万个元素
        const float        a = 2.0f;

        // 1. 在主机上准备数据
        std::vector<float> h_x(N);
        std::vector<float> h_y(N);
        for (int i = 0; i < N; ++i)
        {
            h_x[i] = static_cast<float>(i);
            h_y[i] = static_cast<float>(N - i);
        }

        // 2. 使用我们的 C++ 封装库
        // DeviceVector 的构造函数会自动分配设备内存 (RAII)
        // MyCudaLib::DeviceVector<float> d_x(N);
        // MyCudaLib::DeviceVector<float> d_y(N);

        // // 将数据从主机拷贝到设备
        // d_x.copy_from_host(h_x);
        // d_y.copy_from_host(h_y);

        // // 调用高级算法接口，非常简洁！
        // MyCudaLib::saxpy(a, d_x, d_y);

        // // 将结果拷贝回主机
        // std::vector<float> h_result(N);
        // d_y.copy_to_host(h_result);

        // --- 使用新的异步接口 ---

        // 1. 创建一个 RAII 风格的 Stream 对象
        MyCudaLib::Stream              stream;

        // 2. 分配设备内存
        MyCudaLib::DeviceVector<float> d_x(N);
        MyCudaLib::DeviceVector<float> d_y(N);

        // 3. 在我们创建的流上异步地执行所有操作
        // 这些函数调用会立即返回，GPU 在后台开始工作
        d_x.copy_from_host_async(h_x, stream);
        d_y.copy_from_host_async(h_y, stream);
        MyCudaLib::saxpy(a, d_x, d_y, stream);

        std::vector<float> h_result(N);
        d_y.copy_to_host_async(h_result, stream);

        // 4. ***至关重要的一步***
        // 在访问结果之前，我们必须等待流中的所有操作（拷贝、计算）全部完成
        stream.synchronize();

        // 3. 验证结果
        // 只检查一个元素以作演示
        int   check_index = 123;
        float expected    = a * h_x[check_index] + h_y[check_index];
        if (std::abs(h_result[check_index] - expected) < 1e-5)
        {
            std::cout << "Success! Verification passed." << std::endl;
        }
        else
        {
            std::cout << "Failure! Verification failed." << std::endl;
            std::cout << "Result: " << h_result[check_index] << ", Expected: " << expected << std::endl;
        }
    }
    catch (const MyCudaLib::CudaError& e)
    {
        std::cerr << "A CUDA error occurred: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    // d_x 和 d_y 会在这里自动销毁，它们的析构函数会自动释放设备内存
    return 0;
}