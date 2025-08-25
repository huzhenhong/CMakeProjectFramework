
#include "cuda_algo.hpp"  // 包含我们自己的 C++ 头文件
#include "cuda_algo.h"    // 包含底层的 C ABI 头文件

// 引入 spdlog
// #include "spdlog/spdlog.h"

namespace MyCudaLib
{
    // // --- CudaError 和 checkCuda 的实现 ---
    // CudaError::CudaError(cudaError_t error, const std::string& message)
    //     : std::runtime_error(message + ": " + cudaGetErrorString(error)), errorCode(error) {}

    // cudaError_t CudaError::getErrorCode() const
    // {
    //     return errorCode;
    // }

    // void checkCuda(cudaError_t result)
    // {
    //     if (result != cudaSuccess)
    //     {
    //         spdlog::error("CUDA API call failed with error: {}", cudaGetErrorString(result));
    //         throw CudaError(result, "CUDA API call failed");
    //     }
    // }

    // // --- Stream 类的实现 ---
    // Stream::Stream()
    //     : m_stream(nullptr)
    // {
    //     checkCuda(cudaStreamCreate(&m_stream));
    //     spdlog::debug("CUDA Stream created: {}", fmt::ptr(m_stream));
    // }

    // Stream::~Stream()
    // {
    //     if (m_stream)
    //     {
    //         spdlog::debug("Destroying CUDA Stream: {}", fmt::ptr(m_stream));
    //         // 析构函数中不应抛出异常
    //         cudaStreamDestroy(m_stream);
    //     }
    // }

    // Stream::Stream(Stream&& other) noexcept
    //     : m_stream(other.m_stream)
    // {
    //     other.m_stream = nullptr;
    // }

    // Stream& Stream::operator=(Stream&& other) noexcept
    // {
    //     if (this != &other)
    //     {
    //         if (m_stream)
    //             cudaStreamDestroy(m_stream);
    //         m_stream       = other.m_stream;
    //         other.m_stream = nullptr;
    //     }
    //     return *this;
    // }

    // cudaStream_t Stream::get() const
    // {
    //     return m_stream;
    // }

    // void Stream::synchronize() const
    // {
    //     spdlog::trace("Synchronizing stream: {}", fmt::ptr(m_stream));
    //     checkCuda(cudaStreamSynchronize(m_stream));
    //     spdlog::trace("Stream synchronized: {}", fmt::ptr(m_stream));
    // }

    // --- saxpy 算法的实现 ---
    void saxpy(float a, const DeviceVector<float>& x, DeviceVector<float>& y, const Stream& stream)
    {
        // spdlog::info("Executing SAXPY with a = {}, vector size = {}", a, x.size());
        if (x.size() != y.size())
        {
            // spdlog::error("SAXPY vector size mismatch: x.size()={}, y.size()={}", x.size(), y.size());
            throw std::invalid_argument("Input vectors must have the same size.");
        }

        // 调用底层的异步 C 接口
        checkCuda(saxpy_c_async(a, x.data(), y.data(), static_cast<int>(x.size()), stream.get()));
        // spdlog::debug("SAXPY kernel launched on stream {}", fmt::ptr(stream.get()));
    }

    // 注意：模板类 DeviceVector 的实现依然在 .hpp 文件中

}  // namespace MyCudaLib