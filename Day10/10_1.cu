// manage the threads in a block
#include <stdio.h>
#include "../tools/common.cuh"

__device__ int d_y[4];
__device__ int d_x = 1;
__global__ void visit_global_variable() {
    d_y[0] = d_x + 1;
    d_y[1] = d_x + 2;
    d_y[2] = d_x + 3;
    d_y[3] = d_x + 4;
    printf("d_y[0] = %d\n",d_y[0]);
}
int main(){
    setDevice();
    //init the memory
    int h_y[4] = {0,1,2,3};
    ErrorCheck(cudaMemcpyToSymbol(d_y,h_y,4*sizeof(int)),__FILE__,__LINE__);
    dim3 block(1);
    dim3 grid(1);
    visit_global_variable<<<grid, block>>>();
    cudaDeviceSynchronize();
    ErrorCheck(cudaMemcpyFromSymbol(h_y,d_y,4*sizeof(int)),__FILE__,__LINE__);
    for(int i=0;i<4;i++){
        printf("x[%d] = %d\n",i,h_y[i]);// device vari d_y cannot be visited by host
    }
    cudaDeviceReset();
    return 0;

}