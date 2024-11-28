//a simple example of creating an adding kernel func
#include <stdio.h>
#include "../tools/common.cuh"

__device__  float add(float a, float b) {
    return a + b;
}
void add_with_cpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// N: the number of elements in the array
__global__ void add_with_gpu(float *A, float *B, float *C, int N) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdm = blockDim.x;
    
    int idx = bid * bdm + tid;
    if (idx < N)
        C[idx] = add(A[idx], B[idx]);
    else return;
}

int main(void) {
    setDevice();
    //init the device memory set
    // DeviceMemory dev_mem = setMemory_Init(512, 666);
    FullMemory ful_mem = setMemory_Init(512, 666);
    DeviceMemory dev_mem = ful_mem.deviceMemory;
    DeviceMemory host_mem = ful_mem.hostMemory;

    float *d_A = dev_mem._A;
    float *d_B = dev_mem._B;
    float *d_C = dev_mem._C;
    float *h_A = host_mem._A;
    float *h_B = host_mem._B;
    float *h_C = host_mem._C;
    //init dim3
    dim3 block(1);
    dim3 grid(1);
    add_with_gpu<<<grid, block>>>(d_A, d_B, d_C, 8);
    //if not using cudaDeviceSynchronize(), the result will be wrong, add will print more than 8 times
    cudaDeviceSynchronize();
    //copy the result from device to host
    cudaMemcpy(h_A, d_A, 512 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, 512 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, 512 * sizeof(float), cudaMemcpyDeviceToHost);
    //print the result
    for (int i = 0; i < 8; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }
    //free the memory
    freeMemory(ful_mem);
    printf("Done\n");
    return 0;
}