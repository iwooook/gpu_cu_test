#include <cstddef>
#include <cstdio>
#include <iostream>
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdlib>

#define CHECK_HIP_ERROR(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define N_THREADS_PER_BLOCK 256

__global__ void MFMAPerformanceKernel(void *p_a, void *p_b, size_t N) {
    using float16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    float *fp_a = (float *)p_a;
    float *fp_b = (float *)p_b;

    float a = fp_a[idx];
    float b = fp_b[idx];
    float16 c = {0};

    for (size_t i = 0; i < N; i++) {    
        // v_mfma_f32_32x32x2_f32
        // FLOPs 4096
        c = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, c, 0, 0, 0);
    }

    for (size_t i = 0; i < 16; i++) {
        fp_a[idx + i] = c[i];
    }

}

__global__ void MFMAPerformanceKernel2(void *p_a, void *p_b, size_t N) {
    using fp16_4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
    using float16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    fp16_4 *fp_a = (fp16_4 *)p_a;
    float *fp_a2 = (float *)p_a;
    fp16_4 *fp_b = (fp16_4 *)p_b;

    fp16_4 a = fp_a[idx];
    fp16_4 b = fp_b[idx];
    float16 c = {0};

    for (size_t i = 0; i < N; i++) {    
        // v_mfma_f32_32x32x8_f16
        // FLOPs 16384
        c = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
    }

    for (size_t i = 0; i < 16; i++) {
        fp_a2[idx + i] = c[i];
    }

}

__global__ void MFMAPerformanceKernel3(void *p_a, void *p_b, size_t N) {
    using int16 = __attribute__((__vector_size__(16 * sizeof(int)))) int;

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    long *lp_a = (long *)p_a;
    long *lp_b = (long *)p_b;

    long a = lp_a[idx];
    long b = lp_b[idx];
    int16 c = {0};

    for (size_t i = 0; i < N; i++) {    
        // v_mfma_i32_32x32x16_i8
        // Ops 32768
        c = __builtin_amdgcn_mfma_i32_32x32x16_i8(a, b, c, 0, 0, 0);
    }

    for (size_t i = 0; i < 16; i++) {
        lp_a[idx + i] = c[i];
    }

}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: ./mfma <num_iter> <num_kernel> <num_cus> <data_type>" << std::endl;
        return 1;
    }

    // Parse input arguments
    size_t num_iter = std::atoi(argv[1]) * 1024;
    size_t num_kernel = std::atoi(argv[2]);
    size_t num_cus = std::atoi(argv[3]);
    size_t data_type = std::atoi(argv[4]);

    std::cout << "num_iter=" << num_iter << ", num_kernel=" << num_kernel << std::endl;

    const int threads_per_block = N_THREADS_PER_BLOCK;
    const int buffer_elems = 1e8;

    // function pointer to kernel
    void (*kernel)(void *, void *, size_t);

    if (data_type == 0) {
        kernel = MFMAPerformanceKernel;
        printf("testing FP32\n");
    } else if (data_type == 1) {
        kernel = MFMAPerformanceKernel2;
        printf("testing FP16\n");
    } else {
        kernel = MFMAPerformanceKernel3;
        printf("testing INT8\n");
    }

    // Allocate host and device memory
    float* h_data = new float[buffer_elems];
    float* d_data_a;
    float* d_data_b;
    CHECK_HIP_ERROR(hipMalloc(&d_data_a, buffer_elems * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_data_b, buffer_elems * sizeof(float)));

    // Initialize host data
    for (size_t i = 0; i < buffer_elems; i++) {
        // FIXME: Initialize data
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(d_data_a, h_data, buffer_elems * sizeof(float), hipMemcpyHostToDevice));

    // Warm-up kernel launch
    std::cout << "Warming up..." << std::endl;
    for (size_t i = 0; i < 5; i++) {
        hipLaunchKernelGGL(kernel, dim3(1), dim3(threads_per_block), 0, 0, d_data_a, d_data_b, num_iter);

    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    std::cout << "Warming up done." << std::endl;

    // Benchmark kernel launch
    std::cout << "Evaluating..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_kernel; i++) {
        hipLaunchKernelGGL(kernel, dim3(1), dim3(threads_per_block), 0, 0, d_data_a, d_data_b, num_iter);
    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Evaluation done." << std::endl;

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    // Calculate TFLOPS
    int num_ops_of_mfma;
    if (data_type == 0) {
        num_ops_of_mfma = 4096;
    } else if (data_type == 1) {
        num_ops_of_mfma = 16384;
    } else {
        num_ops_of_mfma = 32768;
    }
    double num_of_ops = static_cast<double>(1 * threads_per_block / 64 * num_ops_of_mfma * num_iter * num_kernel);
    double flops = num_of_ops / elapsed.count();

    std::cout << "Single CU Performance: " << flops / 1e12 << " TFLOPS" << std::endl;
    std::cout << "Performance: " << flops / 1e12 * num_cus << " TFLOPS" << std::endl;

    // Cleanup
    CHECK_HIP_ERROR(hipFree(d_data_a));
    CHECK_HIP_ERROR(hipFree(d_data_b));
    delete[] h_data;

    return 0;
}
