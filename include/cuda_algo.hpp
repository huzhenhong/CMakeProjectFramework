// include/my_cuda_lib.hpp

#ifndef MY_CUDA_LIB_HPP
#define MY_CUDA_LIB_HPP

#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_algo.h"  // 包含底层的 C ABI 头文件

namespace MyCudaLib
{

    // 1. 自定义异常类，用于封装 CUDA 错误
    class CudaError : public std::runtime_error
    {
      public:
        CudaError(cudaError_t error, const std::string& message)
            : std::runtime_error(message + ": " + cudaGetErrorString(error))
            , errorCode(error)
        {
        }

        cudaError_t getErrorCode() const
        {
            return errorCode;
        }

      private:
        cudaError_t errorCode;
    };

    // 辅助函数，用于检查 CUDA API 调用的返回值
    inline void checkCuda(cudaError_t result)
    {
        if (result != cudaSuccess)
        {
            throw CudaError(result, "CUDA API call failed");
        }
    }

    // 2. RAII 风格的设备内存管理类
    template<typename T>
    class DeviceVector
    {
      public:
        // 构造函数：分配设备内存
        explicit DeviceVector(size_t count)
            : d_ptr(nullptr), m_count(count)
        {
            if (m_count > 0)
            {
                checkCuda(cudaMalloc(&d_ptr, m_count * sizeof(T)));
            }
        }

        // 析构函数：释放设备内存
        ~DeviceVector()
        {
            if (d_ptr)
            {
                cudaFree(d_ptr);
            }
        }

        // 删除拷贝构造和拷贝赋值，防止浅拷贝
        DeviceVector(const DeviceVector&)            = delete;
        DeviceVector& operator=(const DeviceVector&) = delete;

        // 实现移动构造函数
        DeviceVector(DeviceVector&& other) noexcept
            : d_ptr(other.d_ptr), m_count(other.m_count)
        {
            other.d_ptr   = nullptr;
            other.m_count = 0;
        }

        // 实现移动赋值运算符
        DeviceVector& operator=(DeviceVector&& other) noexcept
        {
            if (this != &other)
            {
                if (d_ptr)
                    cudaFree(d_ptr);
                d_ptr         = other.d_ptr;
                m_count       = other.m_count;
                other.d_ptr   = nullptr;
                other.m_count = 0;
            }
            return *this;
        }

        // 从主机拷贝数据到设备
        void copy_from_host(const std::vector<T>& host_vec)
        {
            if (host_vec.size() != m_count)
            {
                throw std::runtime_error("Size mismatch in copy_from_host");
            }
            checkCuda(cudaMemcpy(d_ptr, host_vec.data(), m_count * sizeof(T), cudaMemcpyHostToDevice));
        }

        // 从设备拷贝数据到主机
        void copy_to_host(std::vector<T>& host_vec)
        {
            if (host_vec.size() != m_count)
            {
                host_vec.resize(m_count);
            }
            checkCuda(cudaMemcpy(host_vec.data(), d_ptr, m_count * sizeof(T), cudaMemcpyDeviceToHost));
        }

        // 获取底层设备指针
        T* data()
        {
            return d_ptr;
        }

        const T* data() const
        {
            return d_ptr;
        }

        // 获取元素数量
        size_t size() const
        {
            return m_count;
        }

      private:
        T*     d_ptr;
        size_t m_count;
    };


    // 3. 封装后的 C++ SAXPY 算法接口
    void saxpy(float a, const DeviceVector<float>& x, DeviceVector<float>& y)
    {
        if (x.size() != y.size())
        {
            throw std::invalid_argument("Input vectors must have the same size.");
        }

        // 调用底层的 C 接口，并检查错误
        checkCuda(saxpy_c(a, x.data(), y.data(), static_cast<int>(x.size())));
    }

}  // namespace MyCudaLib

#endif  // MY_CUDA_LIB_HPP