// manage the threads in a block
#include <stdio.h>
#include "../tools/common.cuh"

__global__ void add_by2Dthread(float *A, float *B, float *C, int nx, int ny) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = idx * ny + idy;
    if(idx < nx && idy < ny) {
        C[id] = A[id] + B[id];
    }
}
int main(){
    setDevice();
    //init the memory
    FullMemory mem = setMemory_Init(512,666);
    // float *d_A, *d_B, *d_C;
    float *d_A = mem.deviceMemory._A;
    float *d_B = mem.deviceMemory._B;
    float *d_C = mem.deviceMemory._C;
    //nx = 1<<10
    int nx = 1<<5; //1024 = 2^10
    int ny = 1<<5; //00000001 -> 00100000 = 32
    dim3 block(4,4);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    printf("grid.x = %d, grid.y = %d\n", grid.x, grid.y);
    add_by2Dthread<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    // copy the result back to host
    ErrorCheck(cudaMemcpy(mem.hostMemory._C, d_C, 512, cudaMemcpyDeviceToHost),__FILE__,__LINE__);
    for (int i = 0; i < 10; i++) {
        printf("id: %d,Matrix A:%f,Matrix B:%f,Matrix C: %f \n", i, mem.hostMemory._A[i], mem.hostMemory._B[i], mem.hostMemory._C[i]);
    }
    cudaDeviceSynchronize();
    return 0;

}