#include <stdio.h>
//my arch is 8.0,
__global__ void helloFromGPU(void) {
    printf("Hello World from GPU!\n");
}

int main(void) {
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    // cudaDeviceSynchronize() will force the CPU to wait until the GPU has completed all tasks.
    cudaDeviceSynchronize();
    return 0;
}