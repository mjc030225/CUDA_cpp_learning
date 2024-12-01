#include <iostream>
// #include <cuda_runtime.h>

__global__ void sharedMemoryExample(int *input, int *output, int size) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < size) {
        sharedData[tid] = input[index];
        __syncthreads();

        // Perform some computation using shared memory
        output[index] = sharedData[tid] * 2;
    }
}

int main() {
    const int size = 256;
    const int bytes = size * sizeof(int);

    int h_input[size], h_output[size];
    for (int i = 0; i < size; i++) {
        h_input[i] = i;
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sharedMemoryExample<<<blocks, threads, threads * sizeof(int)>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}