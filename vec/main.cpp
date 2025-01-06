#include <iostream>
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdlib>

using float16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

#define CHECK_HIP_ERROR(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define N_THREADS_PER_BLOCK 256

template <typename T>
__global__ void VectorPerformanceKernel(T *p_a, T *p_b, size_t N) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t tid = threadIdx.x;

    float16 a = p_a[idx];
    float16 b = p_b[idx];
    float16 c = {0};

    for (size_t i = 0; i < N; i++) {    
      c += a + b;
    }

    p_a[idx] = c;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./vec <num_iter> <num_kernel> <num_cus>" << std::endl;
        return 1;
    }

    // Parse input arguments
    size_t num_iter = std::atoi(argv[1]) * 1024;
    size_t num_kernel = std::atoi(argv[2]);
    size_t num_cus = std::atoi(argv[3]);

    std::cout << "num_iter=" << num_iter << ", num_kernel=" << num_kernel << std::endl;

    const int threads_per_block = N_THREADS_PER_BLOCK;
    const int buffer_elems = 1e8;

    // Allocate host and device memory
    float16* h_data = new float16[buffer_elems];
    float16* d_data_a;
    float16* d_data_b;
    CHECK_HIP_ERROR(hipMalloc(&d_data_a, buffer_elems * sizeof(float16)));
    CHECK_HIP_ERROR(hipMalloc(&d_data_b, buffer_elems * sizeof(float16)));

    // Initialize host data
    for (size_t i = 0; i < buffer_elems; i++) {
        // FIXME: Initialize data
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(d_data_a, h_data, buffer_elems * sizeof(float16), hipMemcpyHostToDevice));

    // Warm-up kernel launch
    std::cout << "Warming up..." << std::endl;
    for (size_t i = 0; i < 5; i++) {
      hipLaunchKernelGGL((VectorPerformanceKernel<float16>), dim3(1), dim3(threads_per_block), 
                         0, 0, d_data_a, d_data_b, num_iter);
    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    std::cout << "Warming up done." << std::endl;

    // Benchmark kernel launch
    std::cout << "Evaluating..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_kernel; i++) {
        hipLaunchKernelGGL((VectorPerformanceKernel<float16>), dim3(1), dim3(threads_per_block), 
                         0, 0, d_data_a, d_data_b, num_iter);
    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Evaluation done." << std::endl;

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end - start;

    // Calculate TFLOPS
    const int num_ops_of_vec = 32;
    double num_of_ops = static_cast<double>(1 * threads_per_block * num_ops_of_vec * num_iter * num_kernel);
    double flops = num_of_ops / elapsed.count();

    std::cout << "Single CU Performance: " << flops / 1e12 << " TFLOPS" << std::endl;
    std::cout << "Performance: " << flops / 1e12 * num_cus << " TFLOPS" << std::endl;

    // Cleanup
    CHECK_HIP_ERROR(hipFree(d_data_a));
    CHECK_HIP_ERROR(hipFree(d_data_b));
    delete[] h_data;

    return 0;
}
