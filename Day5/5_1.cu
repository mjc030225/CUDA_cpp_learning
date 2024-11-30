#include <stdio.h>
#include "../tools/common.cuh"
//using a 1D grid and 1D block
__global__ void add_kernel_func(float *A,float *B,float *C,int N)
{
    const int thx = threadIdx.x;
    const int bdx = blockIdx.x;
    const int bdm = blockDim.x;
//     int id = thx + bdx * bdm;
    int idx = bdx * bdm + thx;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
    else return;

}
int main()
{
    setDevice();
    dim3 block(512);
    dim3 grid(1);
    FullMemory mem = setMemory_Init(512,666);
    float *A = mem.deviceMemory._A; 
    float *B = mem.deviceMemory._B;
    float *C = mem.deviceMemory._C;
    int N = 512;
    add_kernel_func<<<grid,block>>>(A,B,C,N);
    ErrorCheck(cudaGetLastError(),__FILE__,__LINE__);
    ErrorCheck(cudaDeviceSynchronize(),__FILE__,__LINE__);
    // cudaDeviceSynchronize();
    freeMemory(mem);
    return 0;
    
}
