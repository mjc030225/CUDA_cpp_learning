// a simple example of ticking time of the kernel function
# include <stdio.h>
# include "../tools/common.cuh"
// also the adding function
__global__ void kernel_add(float *a, float *b, float *c, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        c[i] = a[i] + b[i];
    }
}

int main(){
    //init the device
    setDevice();
    //init the memory
    FullMemory mem = setMemory_Init(512,666);
    float *A = mem.deviceMemory._A;
    float *B = mem.deviceMemory._B;
    float *C = mem.deviceMemory._C;
    int N = 512;
    dim3 block(512);
    dim3 grid(1);
    //init the time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //start the time
    cudaEventRecord(start, 0);
    cudaEventQuery(start);
    kernel_add<<<grid, block>>>(A, B, C, N);
    //stop the time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("The elapsed time is %f ms\n", elapsedTime);
    //free the memory
    freeMemory(mem);

    //init the kernel
    return 0;
}
