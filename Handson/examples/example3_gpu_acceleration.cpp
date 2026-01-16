#include "hpc_toolkit.hpp"
#include <iostream>

int main() {
    std::cout << "=== Example 3: GPU Acceleration Demo ===" << std::endl;

#ifdef __CUDACC__
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device 0: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;

    std::cout << "\n=== Vector Operations on GPU ===" << std::endl;

    auto v1 = Vector<float>::random(1000000, -1.0f, 1.0f);
    auto v2 = Vector<float>::random(1000000, -1.0f, 1.0f);

    std::cout << "Vector size: " << v1.size() << " elements" << std::endl;

    {
        ScopedTimer timer("CPU Vector Addition");
        auto v3_cpu = v1 + v2;
    }

    {
        ScopedTimer timer("GPU Vector Addition");
        try {
            auto v1_copy = v1;
            v1_copy.add_gpu(v2);
        } catch (const std::exception& e) {
            std::cerr << "GPU operation failed: " << e.what() << std::endl;
        }
    }

    std::cout << "\nExample 3 completed!" << std::endl;
#else
    std::cout << "CUDA not available - skipping GPU examples" << std::endl;
#endif

    return 0;
}
